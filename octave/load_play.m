function load_play(file)
    data = load(file);
    player = audioplayer (data, 48000);
    playblocking (player);
endfunction
