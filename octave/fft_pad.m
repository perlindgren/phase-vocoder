clear;

Fs = 48000;
T = 1;
t = linspace(0, T, Fs * T);

F = Fs / 10;

y = sin(F * t * 2 * pi);

figure(1); plot(y);

y_fft = fft(y);

y_fft_s = y_fft(1:Fs / 2);

figure(2); plot(abs(y_fft_s));

y_inv_fft = real(ifft([y_fft_s 0 fliplr(conj(y_fft_s(2:end)))]));

figure(3); plot(y_inv_fft);

y_fft_pad = fft([y linspace(0, 0, Fs)]);

y_fft_pad_s = y_fft_pad(1:Fs);

y_inv_fft_pad = real(ifft([y_fft_pad_s 0 fliplr(conj(y_fft_pad_s(2:end)))]));

figure(4); plot(y_inv_fft_pad);

% y_inv = ifft(y);

% figure(4); plot(y_inv);

wild = [y_fft_s linspace(0, 0, Fs / 2)];
wild_inv = real(ifft([wild 0 fliplr(conj(wild(2:end)))]));

figure(5); plot(wild_inv);
