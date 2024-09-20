fs = 1000;
x = 1:fs;
f = 20;
data = sin(2 * pi * f * x / fs);

fft_data = fft(data);

ifft_data = ifft(fft_data);

% figure(1);
plot(x, data(x), '-xo', x, data(1 + rem(x, fs), '-xr');
% hold on
% plot(x, data(x + 17), '-xr');
% hold off
% title("data");
% legend("legend data");

% figure(2);
% plot(x, ifft_data(x));
% title("ifft_daa");
% legend("legend ifft_data");

% plot(x, data);

% plot(x, ifft_data);
