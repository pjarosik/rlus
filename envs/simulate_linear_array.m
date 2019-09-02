% This implementation bases on script 'sim_img.m' from field's II 'cyst example'.
% 'lockfile' program is required to exclusively lock the work of each thread
% per scanline. In Ubuntu, it's available in procmail package.

function [] = simulate_linear_array(id, input_path, output_path)

    % TODO parametrize?
    f0=3.5e6;                %  Transducer center frequency [Hz]
    fs=100e6;                %  Sampling frequency [Hz]
    c=1540;                  %  Speed of sound [m/s]
    lambda=c/f0;             %  Wavelength [m]
    width=lambda;            %  Width of element
    element_height=5/1000;   %  Height of element [m]
    kerf=0.05/1000;          %  Kerf [m]
    % FOCUS SET ONLY ON START, WILL BE MODIFIED BELOW.
    focus=[0 0 70]/1000;     %  Fixed focal point [m]
    N_elements=192;          %  Number of physical elements
    N_active=64;             %  Number of active elements

    set_sampling(fs);
    xmit_aperture = xdc_linear_array(N_elements, width, element_height, kerf, 1, 10,focus);

    impulse_response = sin(2*pi*f0*(0:1/fs:2/f0));
    impulse_response = impulse_response.*hanning(max(size(impulse_response)))';
    xdc_impulse(xmit_aperture, impulse_response);

    excitation = sin(2*pi*f0*(0:1/fs:2/f0));
    xdc_excitation(xmit_aperture, excitation);

    receive_aperture = xdc_linear_array(N_elements, width, element_height, kerf, 1, 10,focus);
    xdc_impulse(receive_aperture, impulse_response);

                                %  Set the different focal zones for reception
    %  Set the apodization
    apo = hanning(N_active)';
    if ~isfolder(output_path)
        mkdir(output_path);
    end

    input_file = fullfile(input_path, "input.mat");

    path_to_go = fullfile(input_path, strcat("go.", num2str(id)));
    path_to_die = fullfile(input_path, strcat("die.", num2str(id)));
    disp(strcat("Starting worker ", num2str(id)));
    disp(path_to_go);
    while true
        disp(strcat('Worker ', num2str(id), ': waiting for the job...'));
        fclose(fopen(fullfile(input_path, strcat("started.", num2str(id))), 'w'))
        while (~isfile(path_to_go)) && (~isfile(path_to_die))
            pause(1);
        end
        if isfile(path_to_die)
            break;
        elseif isfile(path_to_go)
            disp(strcat('Worker ', num2str(id), ' is now proceeding to new job..'))
            delete(path_to_go);
            tic;
            clear point_positions point_amplitudes z_focus no_lines image_width;
            load(input_file, "point_positions", "point_amplitudes", "z_focus", "no_lines", "image_width");

            focal_zones = [z_focus];
            Nf = max(size(focal_zones));
            focus_times = (focal_zones-10/1000)/c; % TODO why -10/1000?

            %z_focus = 60/1000;                %  Transmit focus, XMIT FOCAL POINT
            %no_lines = 50;                    %  Number of lines in image TODO parametrize
            %image_width = 40/1000;            %  Size of image sector THE WIDTH OF THE ENVIRONMENT
            no_lines = double(no_lines);
            d_x = image_width/no_lines;       %  Increment for image
            disp(['LOG: z_focus=', num2str(z_focus)])
            disp(['LOG: no_lines=', num2str(no_lines)])
            disp(['LOG: image_width=', num2str(image_width)])

            % Determining an output directory for a single input file.
            example_dir_path = fullfile(output_path, "input.mat.rf");
            if ~isfolder(example_dir_path)
                mkdir(example_dir_path);
            end

                                % Do imaging line by line
            for i = 1:no_lines
                filename = fullfile(example_dir_path, strcat("ln", num2str(i), ".mat"));
                lock_filename = strcat(filename, ".lock");
                if ~system(sprintf("lockfile -r 0 %s > /tmp/sim_field2_lockfile.log 2>&1", lock_filename))
                    %disp(strcat("creating line ", num2str(i)));
                                % We are first in locking this scanline.
                    x= -image_width/2 +(i-1)*d_x;
                                % Set the focus for this direction with the proper reference point
                    xdc_center_focus(xmit_aperture, [x 0 0]);
                    xdc_focus(xmit_aperture, 0, [x 0 z_focus]);
                    xdc_center_focus(receive_aperture, [x 0 0]);
                    xdc_focus(receive_aperture, focus_times, [x*ones(Nf,1), zeros(Nf,1), focal_zones]);
                                % Calculate the apodization
                    N_pre = round(x/(width+kerf)+N_elements/2-N_active/2);
                    N_post = N_elements-N_pre-N_active;
                    apo_vector = [zeros(1,N_pre) apo zeros(1,N_post)];
                    xdc_apodization(xmit_aperture, 0, apo_vector);
                    xdc_apodization(receive_aperture, 0, apo_vector);
                                % Calculate the received response
                    [rf_data, tstart] = calc_scat(xmit_aperture, receive_aperture, point_positions, point_amplitudes);
                                %  Store the result
                    save(filename, "i", "rf_data", "tstart");
                % else
                %  disp(['Line ',num2str(i),' is being made by another worker.'])
                end
            end
            toc;
            disp(strcat("Worker ", num2str(id), ' finished the job.'))
            fclose(fopen(fullfile(input_path, strcat("ready.", num2str(id))), 'w'))
        end
    end
    disp(strcat("Killing worker ", num2str(id), '.'))
    xdc_free(xmit_aperture)
    xdc_free(receive_aperture)
end
