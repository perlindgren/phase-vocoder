function load_plot(file)
    data = load(file);
    x = 1:length(data);
    figure;
    plot (x, data(x));
    title (file);
endfunction
