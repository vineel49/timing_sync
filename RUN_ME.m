% Timing offset estimation for OFDM systems
close all
clear all
clc
%---------------- SIMULATION PARAMETERS ------------------------------------
SNR_dB = 40; % SNR per bit in dB (in logarithmic scale)
num_frames = 1*(10^4); % number of frames to be simulated
FFT_len = 128; % length of the FFT/IFFT (#subcarriers)
chan_len = 10; % actual number of channel taps
fade_var_1D = 0.5; % 1D fade variance of the channel impulse response
preamble_len = 64; % length of the preamble (training sequence)
assumed_chan_len = 2*chan_len-1; % L_{hr} assumed channel length
cp_len = assumed_chan_len-1; % length of the cyclic prefix
num_bit = 2*FFT_len; % number of data bits per OFDM frame (overall rate is 2)
SNR = 10^(0.1*SNR_dB); % SNR per bit in linear scale
noise_var_1D = 0.5*2*2*fade_var_1D*chan_len/(2*FFT_len*SNR); % 1D noise variance

%------------------------------------------------------------------------
%                        PREAMBLE GENERATION
preamble_data = randi([0 1],1,2*preamble_len);
preamble_qpsk = 1-2*preamble_data(1:2:end)+1i*(1-2*preamble_data(2:2:end));
% Avg. power of preamble part must be equal to avg. power of data part
preamble_qpsk = sqrt(preamble_len/FFT_len)*preamble_qpsk; % (4) in paper
preamble_qpsk_ifft = ifft(preamble_qpsk);
%--------------------------------------------------------------------------
%                         MATCHED FILTER
matched_filter = conj(fliplr(preamble_qpsk_ifft));       
matched_filter_energy = matched_filter*matched_filter'; 
norm_matched_filter = matched_filter/sqrt(matched_filter_energy); % Matched filter is normalized to unit energy                     

%--------------------------------------------------------------------------
Erasure = 0; % frame erasures initialization
C_Ber = 0; % channel errors
tic()
%--------------------------------------------------------------------------
for frame_cnt = 1:num_frames
%                           TRANSMITTER
%Source
data = randi([0 1],1,num_bit); % data

% QPSK mapping (according to the set partitioning principles)
mod_sig = 1-2*data(1:2:end) + 1i*(1-2*data(2:2:end));

% IFFT operation
T_qpsk_sig = ifft(mod_sig); % T stands for time domain

% inserting cyclic prefix and preamble
T_trans_sig = [preamble_qpsk_ifft T_qpsk_sig(end-cp_len+1:end) T_qpsk_sig]; 
%--------------------------------------------------------------------------
%                            CHANNEL   
% Rayleigh channel
fade_chan = sqrt(fade_var_1D)*randn(1,chan_len) + 1i*sqrt(fade_var_1D)*randn(1,chan_len);     

% AWGN
white_noise = sqrt(noise_var_1D)*randn(1,FFT_len + cp_len + preamble_len + chan_len - 1) ...
    + 1i*sqrt(noise_var_1D)*randn(1,FFT_len + cp_len + preamble_len + chan_len - 1); 

%channel output
Chan_Op = conv(T_trans_sig,fade_chan) + white_noise; % Chan_Op stands for channel output
%--------------------------------------------------------------------------
%                          RECEIVER 
tim_offset_est_vec = Chan_Op(1:preamble_len + assumed_chan_len-1); % a row vector

sync_op = conv(norm_matched_filter,tim_offset_est_vec);
[~,index] = max(abs(sync_op));

% timing offset
Diff_Inx = index - preamble_len ;

if (Diff_Inx >= 0) && (Diff_Inx < chan_len) 
      
% channel dft taking timing offset into account
fade_chan2 = [fade_chan.'; zeros(assumed_chan_len-chan_len,1) ];
fade_chan2 = circshift(fade_chan2,chan_len-1-Diff_Inx);
fade_chan_dft = fft(fade_chan2.',FFT_len);

% FINAL DATA DETECTION
Start_Inx = chan_len + Diff_Inx+preamble_len;
End_Inx = Start_Inx+FFT_len-1;
T_rec_sig = Chan_Op(Start_Inx:End_Inx);
F_rec_sig = fft(T_rec_sig); % FFT output

% ML decoding
QPSK_SYM = zeros(4,FFT_len);
QPSK_SYM(1,:) = (1+1i)*ones(1,FFT_len);
QPSK_SYM(2,:) = (1-1i)*ones(1,FFT_len);
QPSK_SYM(3,:) = (-1+1i)*ones(1,FFT_len);
QPSK_SYM(4,:) = (-1-1i)*ones(1,FFT_len);

Dist = zeros(4,FFT_len);
 Dist(1,:)=abs(F_rec_sig-QPSK_SYM(1,:).*fade_chan_dft).^2;
 Dist(2,:)=abs(F_rec_sig-QPSK_SYM(2,:).*fade_chan_dft).^2;
 Dist(3,:)=abs(F_rec_sig-QPSK_SYM(3,:).*fade_chan_dft).^2;
 Dist(4,:)=abs(F_rec_sig-QPSK_SYM(4,:).*fade_chan_dft).^2;
 
 [minim,index] = min(Dist,[],1);
 QPSK_CONSTELL = [1+1i 1-1i -1+1i -1-1i];
 dec_qpsk_seq = QPSK_CONSTELL(index);

 % demapping symbols to bits
 dec_data = zeros(1,2*FFT_len);
 dec_data(1:2:end) = real(dec_qpsk_seq)<0;
 dec_data(2:2:end) = imag(dec_qpsk_seq)<0;
 
 % Calculating total bit errors
C_Ber = C_Ber + nnz(dec_data-data); 
else
    Erasure = Erasure + 1;
end
end
% frame erasure rate
FER = Erasure/num_frames

% bit error rate
BER = C_Ber/(num_frames*num_bit)
toc()

