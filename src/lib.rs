use nih_plug::prelude::*;
use realfft::num_complex::Complex32;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::f32;
use std::sync::Arc;

// Must have BLOCK_SIZE < FFT_SIZE and WINDOW_SIZE divide BLOCK_SIZE
const BLOCK_SIZE: usize = 2048;
const FFT_SIZE: usize = 8192;
const WINDOW_SIZE: usize = 512;

struct PhaseVocoder {
    params: Arc<PhaseVocoderParams>,
    stft: util::StftHelper,

    r2c: Arc<dyn RealToComplex<f32>>,
    c2r: Arc<dyn ComplexToReal<f32>>,
    resample_buffer: Vec<f32>,
    spectrum_buffer: Vec<Complex32>,

    synth_phase: (Vec<f32>, Vec<f32>),
    prev_phase: (Vec<f32>, Vec<f32>),
    reset_phase: (bool, bool),
}

#[derive(Params)]
struct PhaseVocoderParams {
    #[id = "pitch_shift"]
    pub pitch_shift: FloatParam,
}

impl Default for PhaseVocoder {
    fn default() -> Self {
        let mut planner = RealFftPlanner::new();
        let r2c = planner.plan_fft_forward(FFT_SIZE);
        let c2r = planner.plan_fft_inverse(FFT_SIZE);
        let resample_buffer = r2c.make_input_vec();
        let spectrum_buffer = r2c.make_output_vec();

        let phase_len = spectrum_buffer.len();
        let synth_phase = (vec![0.0; phase_len], vec![0.0; phase_len]);
        let prev_phase = (vec![0.0; phase_len], vec![0.0; phase_len]);

        Self {
            params: Arc::new(PhaseVocoderParams::default()),
            stft: util::StftHelper::new(2, BLOCK_SIZE, FFT_SIZE - BLOCK_SIZE),

            r2c,
            c2r,
            resample_buffer,
            spectrum_buffer,

            synth_phase,
            prev_phase,
            reset_phase: (true, true),
        }
    }
}

impl Default for PhaseVocoderParams {
    fn default() -> Self {
        Self {
            pitch_shift: FloatParam::new(
                "Pitch Shift",
                0.0,
                FloatRange::Linear {
                    min: -1200.0,
                    max: 1200.0,
                },
            )
            .with_smoother(SmoothingStyle::Linear(50.0))
            .with_step_size(1.0)
            .with_unit(" cents")
        }
    }
}

impl Plugin for PhaseVocoder {
    const NAME: &'static str = "Phase Vocoder";
    const VENDOR: &'static str = "Robert Fuchs";
    const URL: &'static str = env!("CARGO_PKG_HOMEPAGE");
    const EMAIL: &'static str = "robertfuchsyoshi@gmail.com";

    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(2),
        main_output_channels: NonZeroU32::new(2),

        aux_input_ports: &[],
        aux_output_ports: &[],

        names: PortNames::const_default(),
    }];


    const MIDI_INPUT: MidiConfig = MidiConfig::None;
    const MIDI_OUTPUT: MidiConfig = MidiConfig::None;

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn initialize(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        _buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        context.set_latency_samples(self.stft.latency_samples());

        true
    }

    fn reset(&mut self) {
        self.stft.set_block_size(BLOCK_SIZE);

        self.reset_phase = (true, true);
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let mut pitch_shift = self.params.pitch_shift.value();
        let mut pitch_shift_mult = 2.0_f32.powf(pitch_shift / 1200.0);
        let hann_window = util::window::hann(BLOCK_SIZE);
        let mut synth_window = hann_window.clone();

        self.stft.process_overlap_add(buffer, BLOCK_SIZE / WINDOW_SIZE, |channel_idx, window_buffer| {
            // Since process_overlap_add iterates through all channels before moving onto the next
            // window, we can advance the pitch shift in the block for more accuracy
            if channel_idx == 0 {
                pitch_shift = self.params.pitch_shift.smoothed.next_step(WINDOW_SIZE as u32);
                pitch_shift_mult = 2.0_f32.powf(pitch_shift / 1200.0);

                // Recalculate the synthesis window from the Hann window for smoothing and volume normalization
                // NOTE: Since the real synthesis window depends on the pitch shift of surrounding
                // windows, we calculate an approximation assuming a constant shift
                for i in 0..BLOCK_SIZE {
                    let low_offset_float = (i as f32 - BLOCK_SIZE as f32) / pitch_shift_mult / (WINDOW_SIZE as f32);
                    let high_offset_float = i as f32 / pitch_shift_mult / (WINDOW_SIZE as f32);
                    let mut low_offset = low_offset_float.ceil() as i32;
                    let high_offset = high_offset_float.floor() as i32;
                    if low_offset_float == low_offset_float.ceil() {
                        low_offset += 1
                    }

                    let mut denom_sum = 0.0;
                    for n in low_offset..=high_offset {
                        let offset = (n as f32 * pitch_shift_mult * WINDOW_SIZE as f32) as i32;
                        let offset_idx = i as i32 - offset;
                        denom_sum += hann_window[offset_idx as usize].powi(2);
                    }

                    synth_window[i] = hann_window[i] / denom_sum;
                }
            }

            // Apply Hann Window
            for i in 0..BLOCK_SIZE {
                window_buffer[i] *= hann_window[i];
            }

            self.r2c.process(window_buffer, &mut self.spectrum_buffer).unwrap();

            let magnitude = self.spectrum_buffer.iter().map(|c| c.norm()).collect::<Vec<f32>>();
            let phase = self.spectrum_buffer.iter().map(|c| c.arg()).collect::<Vec<f32>>();
            let synth_phase;
            let prev_phase;
            let reset_phase;
            if channel_idx == 0 {
                synth_phase = &mut self.synth_phase.0;
                prev_phase = &mut self.prev_phase.0;
                reset_phase = &mut self.reset_phase.0;
            }
            else {
                synth_phase = &mut self.synth_phase.1;
                prev_phase = &mut self.prev_phase.1;
                reset_phase = &mut self.reset_phase.1;
            }

            if *reset_phase {
                for i in 0..synth_phase.len() {
                    synth_phase[i] = phase[i];
                }
                *reset_phase = false;
            }
            else {
                // Calculate forward phase using phase vocoder principals
                let phase_derivative = phase.iter().zip(prev_phase.iter()).enumerate().map(|(n, (a, b))| {
                    let diff = a - b - 2.0 * f32::consts::PI * n as f32 * WINDOW_SIZE as f32 / FFT_SIZE as f32;
                    let principal = diff - 2.0 * f32::consts::PI * (diff / 2.0 / f32::consts::PI).round();
                    principal / WINDOW_SIZE as f32 + 2.0 * f32::consts::PI * n as f32 / FFT_SIZE as f32
                }).collect::<Vec<f32>>();

                for i in 0..synth_phase.len() {
                    synth_phase[i] += pitch_shift_mult * WINDOW_SIZE as f32 * phase_derivative[i];
                }
            }

            // Set synth phase of first and last bin if applicable to fix issues with FFT
            synth_phase[0] = 0.0;
            if FFT_SIZE % 2 == 0 {
                let i = synth_phase.len() - 1;
                synth_phase[i] = 0.0;
            }

            for i in 0..prev_phase.len() {
                prev_phase[i] = phase[i];
            }

            // Recalculate the spectrum with our new phases for inverse FFT
            for i in 0..self.spectrum_buffer.len() {
                self.spectrum_buffer[i] = Complex32::from_polar(magnitude[i], synth_phase[i]);
            }

            self.c2r.process(&mut self.spectrum_buffer, &mut self.resample_buffer).unwrap();

            // Apply synthesis window and FFT normalization
            for i in 0..BLOCK_SIZE {
                self.resample_buffer[i] *= synth_window[i];
                self.resample_buffer[i] /= FFT_SIZE as f32;
            }
            // Zero-out all other samples to finish the windowing
            for i in BLOCK_SIZE..FFT_SIZE {
                self.resample_buffer[i] = 0.0;
            }

            // Resample to final buffer
            for i in 0..window_buffer.len() {
                let resample_i = i as f32 * pitch_shift_mult;
                if resample_i.ceil() as usize >= self.resample_buffer.len() {
                    window_buffer[i] = 0.0;
                    continue;
                }

                let floor = resample_i.floor() as usize;
                let ceil = resample_i.ceil() as usize;
                if floor == ceil {
                    window_buffer[i] = self.resample_buffer[floor];
                }
                else {
                    window_buffer[i] = self.resample_buffer[floor] * (resample_i - floor as f32) + self.resample_buffer[ceil] * (ceil as f32 - resample_i);
                }
            }
        });

        ProcessStatus::Normal
    }
}

impl ClapPlugin for PhaseVocoder {
    const CLAP_ID: &'static str = "io.github.thedevelo.phase-vocoder";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("A phase vocoder");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;

    const CLAP_FEATURES: &'static [ClapFeature] = &[ClapFeature::AudioEffect, ClapFeature::PhaseVocoder, ClapFeature::PitchShifter, ClapFeature::Stereo];
}

impl Vst3Plugin for PhaseVocoder {
    const VST3_CLASS_ID: [u8; 16] = *b"PhaseVocoderPSAM";

    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::PitchShift, Vst3SubCategory::Stereo];
}

nih_export_clap!(PhaseVocoder);
nih_export_vst3!(PhaseVocoder);
