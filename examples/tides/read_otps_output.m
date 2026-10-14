function [output_constituents, lon, lat] = read_otps_output ( input_otps_file )
% READ_OTPS_OUTPUT:  reads output of OTPS executable "extract_HC"
%
% The tidal constituents that are extracted can be one of
%         m2, s2, n2, k2, t2, l2, k1, o1, p1, q1, m1, j1, mf, mm, ssa, 
%         m4, and m6
%
% USAGE:  [output_constituents, lon, lat] = read_otps_output ( input_otps_file );
%
% PARAMETERS:
% Input:
%     input_otps_file:
%         ascii text file produces by "extract_HC"
%
%
% Output:
%     output_constituents:
%         array of structures.  Each structure has the following fields:
%             Name
%             Period
%             Amplitude
%             Phase
%         The structure is sorted into descending order with respect to
%         the period.
%     lon, lat:
%         Optionally, the longitude and latitude of the points at which
%         the tidal harmonics are defined.
%
% 


afid = fopen ( input_otps_file, 'r' );
if afid == -1
	msg = sprintf ( '%s:  fopen failed on %s.\n', mfilename, input_otps_file );
	error ( msg );
end

%
% Scan thru it to get the number of lines
line_count = 0;
while 1
	tline = fgetl ( afid );
	if ~ischar ( tline )
		break;
	end
	line_count = line_count + 1;
end




%
% Rewind the file.
fseek(afid,0,-1);




%
% Get past the header
% OTPS produces a header that looks something like the following.
%
%      Model:        EC
%      Elevations (m)
%      Lat     Lon       o1_amp  o1_ph   p1_amp  blah blah blah
fgetl ( afid );
fgetl ( afid );
constituents_line = fgetl ( afid );

%
% figure out what constituents we have
r = constituents_line;
count = 0;
while 1
	[t,r] = strtok ( r );
	if isempty(r)
		break
	end
	index = findstr ( t, 'amp' );
	if index
		count = count + 1;
		constituents{count} = t(1:index-2);
	end

end



num_constituents = length(constituents);

%
% Classify the tidal components.
% for j = 1:num_constituents
% 
% 	%
% 	% what is the period?
% 	switch ( lower(constituents{j}) )
% 
% 	%
% 	% semi-diurnal components
% 	case 'm2'
% 		period(j) = 12.4206;
% 	case 's2'
% 		period(j) = 12.0000;
% 	case 'n2'
% 		period(j) = 12.6583;
% 	case 'k2'
% 		period(j) = 11.97;
% 	case 't2'
% 		period(j) = 12.01;
% 	case 'l2'
% 		period(j) = 12.19;
% 
% 	%
% 	% Diurnal components
% 	case 'k1'
% 		period(j) = 23.9345;
% 	case 'o1'
% 		period(j) = 25.8193;
% 	case 'p1'
% 		period(j) = 24.07;
% 	case 'q1'
% 		period(j) = 26.87;
% 	case 'm1'
% 		period(j) = 24.86;
% 	case 'j1'
% 		period(j) = 23.10;
% 
% 	%
% 	% Long period components
% 	case 'mf'
% 		period(j) = 327.86;
% 	case 'mm'
% 		period(j) = 661.30;
% 	case 'ssa'
% 		period(j) = 2191.43;
% 
% 	case 'm4'
% 		period(j) = 6.21;
% 
% 	case 'm6'
% 		period(j) = 4.14;
% 
% 	otherwise
% 		msg = sprintf ( '%s:  unknown tidal component ''%s''\n', constituents{j} );
% 		error ( msg );
% 	end
% 
% end
% 

%
% allocate the arrays
lat = NaN * zeros ( line_count-3, 1 );
lon = NaN * zeros ( line_count-3, 1 );
amplitude = NaN * zeros ( line_count-3, num_constituents );
phase = NaN * zeros ( line_count-3, num_constituents );

%
% Construct the sscanf format string.  There are two times the number
% of constituents floats, plus the lat and lon.
fmt = repmat ( ' %f', 1, num_constituents*2 );
constituents_fmt = ['%f %f' fmt];

line_count = 3;
while 1
	tline = fgetl ( afid );
	if ~ischar ( tline )
		break;
	end

	line_count = line_count + 1;

	if findstr( 'Site is out of model grid OR land', tline )

		%
		% If the line contains the string "Site is out of model grid OR land", then 
		% skip the constituents.  But we still fill in the lat and lon.
		[d,token_count] = sscanf ( tline, '%f %f %s' );
		if ( token_count ~= 3 )
			msg = sprintf ( '%s:  sscanf failed on line %d in input file %s.\n', mfilename, line_count, input_otps_file );
			error ( msg );
		end
		lat(line_count-3) = d(1);
		lon(line_count-3) = d(2);

	elseif findstr( 'Model:', tline )

		%
		% do nothing
		;

	elseif findstr( 'Lat limits:', tline )

		%
		% do nothing
		;

	elseif findstr( 'Lon limits:', tline )

		%
		% do nothing
		;

	elseif findstr( 'Elevations (m)', tline )

		%
		% do nothing
		;

	elseif findstr( 'Lat     Lon', tline )

		%
		% do nothing
		;

	else

		%
		% We must be over water.  Get all the constituents as well as the lat/lon
		[d,token_count] = sscanf ( tline, constituents_fmt );
		if ( token_count ~= (num_constituents*2 + 2) )
			error_fmt = '%s:  sscanf failed on line %d in input file %s, reading %d inputs instead of %d.\n'
			msg = sprintf ( error_fmt, mfilename, line_count, input_otps_file, token_count, (num_constituents*2 + 2) );
			error ( msg );
		end

		lat(line_count-3) = d(1);
		lon(line_count-3) = d(2);

		if ( d(1) == 0 ) & (d(2) == 0 ) 
			line_count
		end

		point_amplitude = d(3:2:end);
		point_phase = d(4:2:end);
		amplitude(line_count-3,:) = point_amplitude';
		phase(line_count-3,:) = point_phase';

	end

end

fclose ( afid );



%
% Load the output structure with the proper components
for j = 1:num_constituents
	output_constituents(j) = struct ( 'Name', constituents{j}, 'Amplitude', amplitude(:,j), 'Phase', phase(:,j), 'Period', nan );
end


% %
% % sort the periods into descending order, use that to sort the output into
% % descending order with regards to the period.
% %[dud, I] = sort ( period, 'descend' );
% [dud, I] = sort ( period);% edited for a different
% I=flipud(I);% version of sort
% dud=flipud(dud);
% 
% output_constituents = output_constituents(I);
