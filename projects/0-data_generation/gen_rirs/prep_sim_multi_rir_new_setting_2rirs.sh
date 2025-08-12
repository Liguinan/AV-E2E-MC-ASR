#!/bin/bash
# Copyright 2016  Tom Ko
# Apache 2.0
# Script to simulate room impulse responses (RIRs)
# This script uses the image method based on an implementation
# by Habets et al to simulate the room impulse responses (RIRs).
# It samples the room parameters and receiver position in the room
# and then randomly generate a number of RIRs according to different speaker positions.
# To use this script you need to have MATLAB installed and
# download the RIR generator from
# https://github.com/ehabets/RIR-Generator/blob/master/rir_generator.cpp
# Usage: prep_sim_rirs.sh <output-dir>
# E.g. prep_sim_rirs.sh data/simulated_rir_largeroom
sampling_rate=16000    # Sampling rate of the output RIR waveform
output_bit=16          # bits per sample in the RIR waveform
num_room=400           # number of rooms to be sampled
rir_per_room=50       # number of RIR to be sampled for each room
prefix="normal-"       # prefix to the RIR id
room_lower_bound=4    # lower bound of the room length and width
room_upper_bound=10    # upper bound of the room length and width
#rir_duration=0.5         # duration of the output RIR waveform in secs  # change to RT60 length
                       # the smaller the room is, the faster the RIR stabilizes
                       # one should set a smaller duration for small room configs
                       # to avoid getting a lot of zeros at the tail of the waveform
                       # and speed up the generation process
                       # e.g. for small room 1~10m -> 0.5 sec, medium room 10~30m -> 1s , large room 30m~50m -> 2s
generator="/project_bdda7/bdda/gnli/Data/lrs2/sim_rir/RIR-Generator/rir_generator.cpp"   # path to the RIR generator
if [ $# != 1 ]; then
  echo "Usage: "
  echo "  $0 [options] <output-dir>"
  echo "e.g.:"
  echo " $0  data/simulated_rirs"
  exit 1;
fi

output_dir=$1
mkdir -p $output_dir

cat >./genrir.m <<EOF
mex $generator
c = 340;                          % Sound velocity (m/s)
array_width = 0.28                % 15 mic, the middle mic is the reference mic and both width are 0.28cm
fs = $sampling_rate;              % Sample frequency (samples/s)
num_sample = fs * $rir_duration;  % Number of samples
mtype = 'omnidirectional';        % Type of microphone
order = 10;                       % Reflection order
dim = 3;                          % Room dimension
orientation = 0;                  % Microphone orientation (rad)
hp_filter = 0;                    % Disable high-pass filter
BitsPerSample = $output_bit;      % Bits per sample in the RIR waveform
Room_bound_x = [$room_lower_bound $room_upper_bound];   % range for sampling the length of the room (m)
Room_bound_y = [$room_lower_bound $room_upper_bound];   % range for sampling the width of the room
Room_bound_z = [3 6];                 % range for sampling the height of the room
Absorption_bound = [0.2 0.8];       % range for sampling the absorption coefficient
                                      % The absorption coefficient of a material is a number between 0 and 1
                                      % E.g. for a material which offer no reflection, it would have
                                      % an absorption coefficient very close to 1
SMD_bound = [1 5];                    % allowed range for speaker microphone distances (SMDs)
output_dir = '$output_dir';
RIRset_prefix = '$prefix';
file_room_info = fopen(strcat(output_dir, '/', 'room_info'),'w');
file_rirlist = fopen(strcat(output_dir, '/', 'rir_list'),'w');
file_rirlist2 = fopen(strcat(output_dir, '/', 'rir_list2'),'w');

for room_id = 1 : $num_room
    room_name = strcat('Room', sprintf('%03d',room_id));
    room_dir = strcat(output_dir, '/', room_name);
    if ~exist(room_dir)
        mkdir(room_dir);
    end    

    % Sample the room size
    room_x = round(rand * (Room_bound_x(2)-Room_bound_x(1)) + Room_bound_x(1), 2);
    room_y = round(rand * (Room_bound_y(2)-Room_bound_y(1)) + Room_bound_y(1), 2);
    room_z = round(rand * (Room_bound_z(2)-Room_bound_z(1)) + Room_bound_z(1), 2);
    Room_xyz = [room_x room_y room_z];          % Room dimensions [x y z]
    Volume = Room_xyz(1) * Room_xyz(2) * Room_xyz(3);
    Surface = 2 * (Room_xyz(1) * Room_xyz(2) + Room_xyz(2) * Room_xyz(3) + Room_xyz(3) * Room_xyz(1));

    % Sample the absorption coefficient
    absorption = round(rand * (Absorption_bound(2)-Absorption_bound(1)) + Absorption_bound(1), 2);
    reflection = sqrt(1-absorption);
    % Here we assume all the walls of a room are built by the same material, and therefore share the same absorption coefficient
    Reflect = [reflection reflection reflection reflection reflection reflection];

    % Sample the microphone position
    while 1
        Ref_Mic_xyz = [rand*Room_xyz(1) rand*Room_xyz(2) rand*Room_xyz(3)];    % Receiver position [x y z]
        %Check if the linear array is within the correct room range, otherwise resample
        if (Ref_Mic_xyz(1) + array_width < Room_xyz(1)) & (Ref_Mic_xyz(1) - array_width > 0)
            break
        end
    end      
    Mic_xyz = [Ref_Mic_xyz(1) - 0.28, Ref_Mic_xyz(2), Ref_Mic_xyz(3);
        Ref_Mic_xyz(1) - 0.21, Ref_Mic_xyz(2), Ref_Mic_xyz(3);
        Ref_Mic_xyz(1) - 0.15, Ref_Mic_xyz(2), Ref_Mic_xyz(3);
        Ref_Mic_xyz(1) - 0.10, Ref_Mic_xyz(2), Ref_Mic_xyz(3);
        Ref_Mic_xyz(1) - 0.06, Ref_Mic_xyz(2), Ref_Mic_xyz(3);
        Ref_Mic_xyz(1) - 0.03, Ref_Mic_xyz(2), Ref_Mic_xyz(3);
        Ref_Mic_xyz(1) - 0.01, Ref_Mic_xyz(2), Ref_Mic_xyz(3);
        Ref_Mic_xyz(1), Ref_Mic_xyz(2), Ref_Mic_xyz(3);
        Ref_Mic_xyz(1) + 0.01, Ref_Mic_xyz(2), Ref_Mic_xyz(3);
        Ref_Mic_xyz(1) + 0.03, Ref_Mic_xyz(2), Ref_Mic_xyz(3);
        Ref_Mic_xyz(1) + 0.06, Ref_Mic_xyz(2), Ref_Mic_xyz(3);
        Ref_Mic_xyz(1) + 0.10, Ref_Mic_xyz(2), Ref_Mic_xyz(3);
        Ref_Mic_xyz(1) + 0.15, Ref_Mic_xyz(2), Ref_Mic_xyz(3);
        Ref_Mic_xyz(1) + 0.21, Ref_Mic_xyz(2), Ref_Mic_xyz(3);
        Ref_Mic_xyz(1) + 0.28, Ref_Mic_xyz(2), Ref_Mic_xyz(3);]
    
    %calculate RT60 according the specified room and reflection
    alpha = 1 - reflection^2;
    V = Room_xyz(1) * Room_xyz(2) * Room_xyz(3);
    S = 2 * (Room_xyz(1) * Room_xyz(2) + Room_xyz(2) * Room_xyz(3) + Room_xyz(1) * Room_xyz(3));
    RT60 = 24 * V * log(10.0) / (c * S * alpha)

    fprintf(file_room_info, '%s%s %3.2f %3.2f %3.2f %3.2f %3.2f %3.2f %3.2f\n', RIRset_prefix, room_name, Room_xyz, Ref_Mic_xyz, absorption);

    for rir_id = 1 : $rir_per_room
        rir_id
        while 1
            % Sample a point within the sphere
            elevation = 0;
            azimuth = pi*rand;
            radii = SMD_bound(2) * (rand.^(1/3));
            [offset_x, offset_y, offset_z] = sph2cart(azimuth,elevation,radii);
            Offset_xyz = [offset_x offset_y offset_z];
            % Source_xyz = Offset_xyz + Ref_Mic_xyz;
            Source_xyz = Offset_xyz + [Ref_Mic_xyz(1) - 0.28, Ref_Mic_xyz(2), Ref_Mic_xyz(3)];
            % Check if the source position is within the correct range, otherwise resample
            if Source_xyz <= Room_xyz & Source_xyz >= 0
                break
            end
        end
        % [rir] = rir_generator(c, fs, Mic_xyz, Source_xyz, Room_xyz, Reflect, num_sample, mtype, order, dim, orientation, hp_filter);
        [rir] = rir_generator(c, fs, Mic_xyz, Source_xyz, Room_xyz, Reflect, fs * RT60, mtype, order, dim, orientation, hp_filter);
        [rir2] = rir_generator(c, fs, Mic_xyz, Source_xyz, Room_xyz, 0, fs * RT60, mtype, order, dim, orientation, hp_filter);
        rir_name = strcat(room_name, '-', sprintf('%05d',rir_id));
        rir_filename = strcat(room_dir, '/', rir_name, '.wav');
        rir2_filename = strcat(room_dir, '/', rir_name, '_2', '.wav');
        audiowrite(rir_filename, rir', fs, 'BitsPerSample', BitsPerSample);
        audiowrite(rir2_filename, rir2', fs, 'BitsPerSample', BitsPerSample);
        
        rir_info = horzcat('--rir-id ', RIRset_prefix, rir_name, ' --room-id ', RIRset_prefix, room_name, ' --rt60 ', sprintf('%3.2f', RT60), ' --azimuth ', sprintf('%3.2f', rad2deg(azimuth)), ' --smd ', sprintf('%3.2f', norm(Offset_xyz, 2)),' ', rir_filename);
        rir2_info = horzcat('--rir-id ', RIRset_prefix, rir_name, ' --room-id ', RIRset_prefix, room_name, ' --rt60 ', sprintf('%3.2f', 0), ' --azimuth ', sprintf('%3.2f', rad2deg(azimuth)), ' --smd ', sprintf('%3.2f', norm(Offset_xyz, 2)),' ', rir2_filename);
        fprintf(file_rirlist, '%s\n', rir_info);
        fprintf(file_rirlist2, '%s\n', rir2_info);
    end
end
fclose(file_room_info);
fclose(file_rirlist);
EOF
matlab -nosplash -nodesktop < ./genrir.m
rm genrir.m rir_generator.mexa64

