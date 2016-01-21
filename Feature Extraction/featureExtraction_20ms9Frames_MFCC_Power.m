%filename = 'C:\Users\giova\Documents\Università\Magistrale\Apprendimento Automatico\media\paolo\1990-11-04-04-36-02-00\timit\train\dr1\fcjf0\sa2.wav';
%[y,Fs] = audioread(filename);
%[cep2, spec2] = rastaplp(y, Fs, 1, 13);

%filename = 'E:\timit\train\dr8\mtcs0\sx82.wav';
%[s,fs,wrd,phn]=readsph(filename,'wt');

training_set = zeros(1, 477);
training_labels = {};

number_of_windows_plp = [];
number_of_samples = [];

for i=1:8                           % Navigate the folders (dr1 ... dr8)
    filename = strcat('C:\timit\train\dr',num2str(i));
    cd(filename);
    directories = dir;
    num_directories = length(directories)-2;
    for j=1:num_directories
        actual_directory = directories(j+2).name; % +2 because the first two fields are "." and ".."
        path = strcat(filename,'\',actual_directory);
        cd(path);
        files_in_path = dir;
        num_files = length(files_in_path)-2;
        for z=1:num_files % cycle trhough all the wav files
            actual_file = files_in_path(z+2).name;
            if strfind(actual_file, 'wav') ~= 0
                wav_file_path = strcat(path, '\', actual_file); % get the path of the wav file
                
                [training, labels] = getPhoneme_20ms9Frames_MFCC_Power(wav_file_path);
                training_set = [training_set; training];
                training_labels = [training_labels; labels];
            end
        end
    end
end