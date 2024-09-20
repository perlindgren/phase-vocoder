fs = 1000;
x = 1:fs;
x2 = 1:fs * 2;
f = 20;
data = sin(2 * pi * f * x / fs);

fft_data = fft(data);

fft_data_stretch = fft_data;

for i = 1:fs
    fft_data_stretch(i) = fft_data(round (0.5 + i / 2));
endfor

ifft_data = ifft(fft_data, fs);

ifft_data2 = ifft(fft_data_stretch, fs * 2);

% figure(1);
plot(x, ifft_data(x), '-b', x2, ifft_data2(x2), '-r');
%plot(data, '-b', ifft_data, '-r');
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
