function load_plot_fft(file)
    data = load(file);
    fft_data = fft(data);
    printf("hello %d", length(data));
    x = 1:length(fft_data);
    figure;
    plot (x, abs(fft_data(x)) / (length(data) * length(data)));
    title(file);
endfunction
