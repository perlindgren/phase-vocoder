fs = 48000; % sample frequncy
x = 1:fs; % sample size

f_ot = 8; % third overton

f_sch = 82.41 * f_ot; % frequency to analyze, low E guitarr

p = 20; % number of periods
ws = round(p * fs / f_sch); % window size

latency = 0.5 * ws / fs

w = -ws / 2:ws / 2; % windw

f = 82.41 * f_ot;
data = sin(2 * pi * f * x / fs);

f2 = 87.31 * f_ot;
data2 = sin(2 * pi * f2 * x / fs);

% complex representation
sin_cos = sin(2 * pi * f_sch * w / fs) + j * cos(2 * pi * f_sch * w / fs);

han = hanning(ws + 1)';
sin_cos_han = (sin_cos .* han); % element by element

c = conv(data, sin_cos_han);
c2 = conv(data2, sin_cos_han);

figure(1);
clf;
hold on;
plot(data, '-b');
plot(data2, '-r');
hold off;

figure(2);
clf;
hold on;
plot(w, abs(sin_cos), '-b');
plot(w, real(sin_cos), '-r');
plot(w, imag(sin_cos), '-g');
hold off;

figure(3);
clf;
hold on;
plot(w, abs(sin_cos_han), '-b');
plot(w, real(sin_cos_han), '-r');
plot(w, imag(sin_cos_han), '-g');
hold off;

figure(4);
clf;
hold on;
plot(abs(c), '-b');
% plot(real(c), '-r');
% plot(imag(c), '-g');

plot(abs(c2), '-r');
hold off;

% plot(han, '-b');

% figure(2);
% plot(w, sin_cos, '-');

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
