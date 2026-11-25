% GNU Octave / MATLAB parity runner. Requires octave-image for fspecial/histeq.
pkg load image;
input_file = argv(){1};
output_file = argv(){2};
loaded = load(input_file);
img = loaded.image;
kernel = loaded.kernel;
out_same = conv2(img, kernel, 'same');
out_valid = conv2(img, kernel, 'valid');
out_full = conv2(img, kernel, 'full');
gaussian = fspecial('gaussian', [5 5], 1.2);
hist_input = uint8(loaded.hist_input);
hist_output = histeq(hist_input, 256);
save('-mat7-binary', output_file, 'out_same', 'out_valid', 'out_full', 'gaussian', 'hist_output');
