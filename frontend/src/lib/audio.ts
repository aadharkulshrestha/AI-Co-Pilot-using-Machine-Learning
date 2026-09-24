/**
 * Cockpit Audio Synthesis & Voice Co-Pilot
 * Implements Web Audio API oscillator synthesis for realistic Airbus/Boeing warnings
 * and Web Speech API for voice annunciations.
 */

class CockpitAudioSystem {
  private audioCtx: AudioContext | null = null;
  private isMuted: boolean = false;
  private isVoiceMuted: boolean = false;
  private lastAnnouncedText: string = "";
  private lastAnnounceTime: number = 0;

  private getContext(): AudioContext | null {
    if (typeof window === "undefined") return null;
    if (!this.audioCtx) {
      const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
      if (AudioContextClass) {
        this.audioCtx = new AudioContextClass();
      }
    }
    if (this.audioCtx && this.audioCtx.state === "suspended") {
      this.audioCtx.resume();
    }
    return this.audioCtx;
  }

  public toggleMute(): boolean {
    this.isMuted = !this.isMuted;
    return this.isMuted;
  }

  public toggleVoice(): boolean {
    this.isVoiceMuted = !this.isVoiceMuted;
    return this.isVoiceMuted;
  }

  public getMuteState(): { audio: boolean; voice: boolean } {
    return { audio: this.isMuted, voice: this.isVoiceMuted };
  }

  /**
   * Master Warning Sound (Airbus Continuous Repetitive Chime / Triple Beep)
   */
  public playMasterWarning() {
    if (this.isMuted) return;
    const ctx = this.getContext();
    if (!ctx) return;

    const now = ctx.currentTime;
    const freqs = [880, 880, 880]; // A5 triple pulsed beep

    freqs.forEach((freq, idx) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = "sawtooth";
      osc.frequency.setValueAtTime(freq, now + idx * 0.14);

      gain.gain.setValueAtTime(0, now + idx * 0.14);
      gain.gain.linearRampToValueAtTime(0.28, now + idx * 0.14 + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.001, now + idx * 0.14 + 0.11);

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start(now + idx * 0.14);
      osc.stop(now + idx * 0.14 + 0.12);
    });
  }

  /**
   * Master Caution Sound (Single Crisp Two-Tone Chime)
   */
  public playMasterCaution() {
    if (this.isMuted) return;
    const ctx = this.getContext();
    if (!ctx) return;

    const now = ctx.currentTime;
    const osc1 = ctx.createOscillator();
    const osc2 = ctx.createOscillator();
    const gain = ctx.createGain();

    osc1.type = "sine";
    osc2.type = "sine";
    osc1.frequency.setValueAtTime(587.33, now); // D5
    osc2.frequency.setValueAtTime(880.0, now + 0.08); // A5

    gain.gain.setValueAtTime(0, now);
    gain.gain.linearRampToValueAtTime(0.2, now + 0.03);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.5);

    osc1.connect(gain);
    osc2.connect(gain);
    gain.connect(ctx.destination);

    osc1.start(now);
    osc1.stop(now + 0.2);
    osc2.start(now + 0.08);
    osc2.stop(now + 0.5);
  }

  /**
   * Informational Cockpit Ping
   */
  public playInfoPing() {
    if (this.isMuted) return;
    const ctx = this.getContext();
    if (!ctx) return;

    const now = ctx.currentTime;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();

    osc.type = "sine";
    osc.frequency.setValueAtTime(1046.5, now); // C6

    gain.gain.setValueAtTime(0, now);
    gain.gain.linearRampToValueAtTime(0.12, now + 0.02);
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.35);

    osc.connect(gain);
    gain.connect(ctx.destination);

    osc.start(now);
    osc.stop(now + 0.35);
  }

  /**
   * Voice Co-Pilot Speech Synthesizer
   */
  public speak(text: string, force: boolean = false) {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
    if (this.isVoiceMuted && !force) return;

    const now = Date.now();
    // Prevent repeating the same announcement within 8 seconds
    if (!force && text === this.lastAnnouncedText && now - this.lastAnnounceTime < 8000) {
      return;
    }

    window.speechSynthesis.cancel(); // Stop current speech
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.05;
    utterance.pitch = 0.95;
    utterance.volume = 1.0;

    // Pick English Voice if available
    const voices = window.speechSynthesis.getVoices();
    const englishVoice = voices.find((v) => v.lang.startsWith("en") && (v.name.includes("Google") || v.name.includes("Natural") || v.name.includes("David") || v.name.includes("Samantha")));
    if (englishVoice) {
      utterance.voice = englishVoice;
    }

    this.lastAnnouncedText = text;
    this.lastAnnounceTime = now;
    window.speechSynthesis.speak(utterance);
  }
}

export const cockpitAudio = new CockpitAudioSystem();
