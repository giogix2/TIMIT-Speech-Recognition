function [training_set2, training_labels2] = getPhoneme_20ms9Frames_MFCC_Power( path )
% Get the phoneme corresponding on each window returned by the rasta-plp
% function

% List of phonemes corresponding to vowels
vowels = strvcat('iy', 'ih', 'eh', 'ey', 'ae', 'aa', 'aw', 'ay', 'ah', 'ao', 'oy', 'ow', 'uh', 'uw', 'ux', 'er', 'ax', 'ix', 'axr', 'ax-h');

% Read the wav file
[s,fs,wrd,phn]=readsph(path,'wt');
% Get the plp of the entire file (also the delta and second order delta)
% [cep, spec] = rastaplp(s, fs, 0, 12);
cep = PLP(s, 320, 160, 11, fs);
% Get the MFCC features
[cep_MFCC, spec_MFCC] = melfcc(s, fs, 'wintime', 0.020);
del = deltas(cep);
ddel = deltas(deltas(cep,5),5);
cep_trasp = cep';
del_trasp = del';
ddel_trasp = ddel';
cep_MFCC_trasp = cep_MFCC';

% Get the energy
[y, energy] = powspec(s, fs, 0.020);
energy = energy';

% Put PLP, delta and second order delta in one matrix
training_set = [cep_trasp cep_MFCC_trasp del_trasp ddel_trasp energy];
% Get the number of plp windows (each window is large 400 samples and is moved 160 samples per time. Overlapping = 240)
number_of_windows = size(cep);
number_of_windows = number_of_windows(2);

% Initialize the labels as a Matlab cell structure
training_labels = {};

last_phoneme_num = 1;
for q=1:number_of_windows

    sample_num_begin = 160*(q-1); % number of the first sample composing the window
    sample_num_end = sample_num_begin+400; % number of the last sample composing the window
    sample_num_middle = (sample_num_begin+sample_num_end)/2; % number of the sample in the middle of the window
    
    % If the final symbol 'h#' comes much erlier than the end of the wav.
    % (there are other PLP frames after the 'h#')

    if sample_num_middle>phn{length(phn),1}(2)
        training_labels = [training_labels, 'h#'];
    end
    % If the first phoneme, which is the symbol 'h#', doesn't start from
    % zero. In this case there are PLP frames preceding the beginning of
    % the sound
    if sample_num_middle<=phn{1,1}(1) && phn{1,1}(1)~=0
        training_labels = [training_labels, 'h#'];
    end
    
    for phoneme_num=last_phoneme_num:length(phn)
        phoneme_begin_end = phn{phoneme_num,1};
        phoneme_begin = phoneme_begin_end(1);
        phoneme_end = phoneme_begin_end(2);
        if phoneme_begin<sample_num_middle && phoneme_end>=sample_num_middle
            phoneme = phn{phoneme_num,2};
            training_labels = [training_labels, phoneme];
            last_phoneme_num = phoneme_num;
        end
    end
end
training_labels = training_labels';

% ================CONCATENATE FRAMES (7 FRAMES FOR EACH PHONEME)===========

training_set2 = [];
training_labels2 = {};
for q=1:number_of_windows
    % Consider just the vowels
    if ~(isempty(strmatch(training_labels{q}, vowels, 'exact')))
        if q>4 && q<number_of_windows-4
            feature_vector = [training_set(q-4,:) training_set(q-3,:) training_set(q-2,:) training_set(q-1,:) training_set(q,:) training_set(q+1,:) training_set(q+2,:) training_set(q+3,:) training_set(q+4,:)];
            training_set2 = [training_set2; feature_vector];
            training_labels2 = [training_labels2, training_labels{q}];
            a = 0;
        elseif q == 4
            feature_vector = [zeros(1,53) training_set(q-3,:) training_set(q-2,:) training_set(q-1,:) training_set(q,:) training_set(q+1,:) training_set(q+2,:) training_set(q+3,:) training_set(q+4,:)];
            training_set2 = [training_set2; feature_vector];
            training_labels2 = [training_labels2, training_labels{q}];
        elseif q == 3
            feature_vector = [zeros(1,53) zeros(1,53) training_set(q-2,:) training_set(q-1,:) training_set(q,:) training_set(q+1,:) training_set(q+2,:) training_set(q+3,:) training_set(q+4,:)];
            training_set2 = [training_set2; feature_vector];
            training_labels2 = [training_labels2, training_labels{q}];
        elseif q == 2
            feature_vector = [zeros(1,53) zeros(1,53) zeros(1,53) training_set(q-1,:) training_set(q,:) training_set(q+1,:) training_set(q+2,:) training_set(q+3,:) training_set(q+4,:)];
            training_set2 = [training_set2; feature_vector];
            training_labels2 = [training_labels2, training_labels{q}];
        elseif q == 1
            feature_vector = [zeros(1,53) zeros(1,53) zeros(1,53) zeros(1,53) training_set(q,:) training_set(q+1,:) training_set(q+2,:) training_set(q+3,:) training_set(q+4,:)];
            training_set2 = [training_set2; feature_vector];
            training_labels2 = [training_labels2, training_labels{q}];
        elseif q == number_of_windows-4
            feature_vector = [training_set(q-4,:) training_set(q-3,:) training_set(q-2,:) training_set(q-1,:) training_set(q,:) training_set(q+1,:) training_set(q+2,:) training_set(q+3,:) zeros(1,53)];
            training_set2 = [training_set2; feature_vector];
            training_labels2 = [training_labels2, training_labels{q}];
        elseif q == number_of_windows-3
            feature_vector = [training_set(q-4,:) training_set(q-3,:) training_set(q-2,:) training_set(q-1,:) training_set(q,:) training_set(q+1,:) training_set(q+2,:) zeros(1,53) zeros(1,53)];
            training_set2 = [training_set2; feature_vector];
            training_labels2 = [training_labels2, training_labels{q}]; 
        elseif q == number_of_windows-2
            feature_vector = [training_set(q-4,:) training_set(q-3,:) training_set(q-2,:) training_set(q-1,:) training_set(q,:) training_set(q+1,:) zeros(1,53) zeros(1,53) zeros(1,53)];
            training_set2 = [training_set2; feature_vector];
            training_labels2 = [training_labels2, training_labels{q}];
        elseif q == number_of_windows-1
            feature_vector = [training_set(q-4,:) training_set(q-3,:) training_set(q-2,:) training_set(q-1,:) training_set(q,:) zeros(1,53) zeros(1,53) zeros(1,53) zeros(1,53)];
            training_set2 = [training_set2; feature_vector];
            training_labels2 = [training_labels2, training_labels{q}];
        end
        
    end
end

training_labels2 = training_labels2';
end

