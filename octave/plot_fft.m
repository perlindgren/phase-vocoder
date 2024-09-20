fs = 1000;
x = 1:fs;
f = 20;
data = sin(2 * pi * f * x / fs);

fft_data = fft(data);

plot(x, fft_data);
