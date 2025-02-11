clear;

Fs = 48000;
T = 1;
t = linspace(0, T, Fs * T);
F1 = 82;
F2 = 82.3;


r = (1:1000);

y1 = sin(2*pi*F1*t); 
y2 = sin(2*pi*F2*t);
y1r = y1(r);
y2r = y2(r);

figure(1); clf; hold on;
plot(y1); plot(y2);

y1f = fft(y1)/Fs;
y2f = fft(y2)/Fs;

figure(2); clf; hold on;
plot(abs(y1f)); plot(abs(y2f));

y1fr = fft(y1r, Fs*10) / Fs; 
y2fr = fft(y2r, Fs*10) / Fs;

figure(3); clf; hold on;
plot(abs(y1fr)); plot(abs(y2fr));

