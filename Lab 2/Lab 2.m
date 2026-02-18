%%
clc; clear;

%% Thermocouple Calibration Data Visualization
% This script plots voltage and temperature over time for three substances.

% 1. Load or Define your data
% Replace 'simout_LN2', etc., with your actual Simulink export variable names.
% Structure expected: .time (vector) and .data (vector)
acetone = load("acetone.mat").out;
ice = load("icewater.mat").out;
ln2 = load("n2.mat").out;


% Example Data Placeholder (Uncomment if running without workspace data)
% t = 0:0.1:10;
% ln2_v = -5.0 + randn(size(t))*0.01;  % Voltage
% ice_v = 0.0 + randn(size(t))*0.01;
% acetone_v = -3.5 + randn(size(t))*0.01;
% (Repeat similar logic for temperature)

%% 2. Create Figure
figure('Color', 'w', 'Name', 'Thermocouple Calibration Analysis');

% --- Subplot 1: Voltage Over Time ---
subplot(2, 1, 1);
hold on; grid on;
plot(ln2.voltage_data, 'b', 'LineWidth', 1.5, 'DisplayName', 'Liquid Nitrogen (~77K)');
plot(ice.voltage_data, 'c', 'LineWidth', 1.5, 'DisplayName', 'Ice Water (~273K)');
plot(acetone.voltage_data, 'm', 'LineWidth', 1.5, 'DisplayName', 'Acetone');

title('Thermocouple Amplifier Output (Voltage)');
xlabel('Time (s)');
ylabel('Voltage (mV)');
legend('Location', 'best');

% --- Subplot 2: Temperature Over Time ---
subplot(2, 1, 2);
hold on; grid on;
plot(ln2.temp_data, 'b', 'LineWidth', 1.5, 'DisplayName', 'Liquid Nitrogen');
plot(ice.temp_data, 'c', 'LineWidth', 1.5, 'DisplayName', 'Ice Water');
plot(acetone.temp_data, 'm', 'LineWidth', 1.5, 'DisplayName', 'Acetone');

title('Calculated Temperature (Conversion Polynomial)');
xlabel('Time (s)');
ylabel('Temperature (°C)');
legend('Location', 'best');

sgtitle('Thermocouple Calibration Results'); % Super title

%% Thermocouple Polynomial Re-Calibration Script
% This script fits the lowest three terms (a*V^2 + b*V + c) 
% while keeping the higher-order terms fixed.

% --- 1. Measured Data (INPUT YOUR SIMULINK VALUES HERE) ---
% Example: Voltages recorded from your amplifier
V_ice = mean(ice.voltage_data.Data);      % Measured voltage in ice water (mV)
V_LN2 = mean(ln2.voltage_data.Data);     % Measured voltage in liquid nitrogen (mV)
V_acetone = mean(acetone.voltage_data.Data); % Measured voltage in acetone (mV)

% Known reference temperatures (Celsius)
T_ice = 0;
T_LN2 = -195.8;    % Standard boiling point of LN2
T_acetone = -80;  % Standard freezing point of Acetone (adjust if using dry ice -78)

% Combine into vectors
V_meas = [V_ice; V_LN2; V_acetone];
T_meas = [T_ice; T_LN2; T_acetone];

% --- 2. Original Fixed Coefficients (V^6 to V^3) ---
c6 = -0.0090;
c5 = 0.151;
c4 = -1.040;
c3 = 4.149;

% --- 3. Calculate Residuals ---
% We want: T = (c6*V^6 + c5*V^5 + c4*V^4 + c3*V^3) + (a*V^2 + b*V + c)
% Let Y = T - (c6*V^6 + c5*V^5 + c4*V^4 + c3*V^3)
% Then we solve Y = a*V^2 + b*V + c

P_high = @(V) c6*V.^6 + c5*V.^5 + c4*V.^4 + c3*V.^3;
Y = T_meas - P_high(V_meas);

% --- 4. Solve for a, b, c ---
% We solve the system: [V^2, V, 1] * [a; b; c] = Y
A_matrix = [V_meas.^2, V_meas, ones(3,1)];
coeffs_low = A_matrix \ Y; % Linear solver

a_new = coeffs_low(1);
b_new = coeffs_low(2);
c_new = coeffs_low(3);

% --- 5. Display Results ---
fprintf('--- Updated Polynomial Coefficients ---\n');
fprintf('V^2 term (a): %.4f (Old: -13.436)\n', a_new);
fprintf('V^1 term (b): %.4f (Old: 105.278)\n', b_new);
fprintf('V^0 term (c): %.4f (Old: 0.1426)\n', c_new);

% --- 6. Verification Plot ---
V_plot = linspace(min(V_meas)-1, max(V_meas)+1, 100);
T_new = P_high(V_plot) + a_new*V_plot.^2 + b_new*V_plot + c_new;

figure('Color', 'w');
plot(V_plot, T_new, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Updated Model');
hold on;
plot(V_meas, T_meas, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r', 'DisplayName', 'Calibration Points');
grid on; xlabel('Voltage (mV)'); ylabel('Temperature (°C)');
title('Recalibrated Thermocouple Polynomial');
legend('Location', 'best');