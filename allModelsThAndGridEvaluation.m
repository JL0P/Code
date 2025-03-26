%% =========================
% MATLAB SCRIPT: Multi Threshold Evaluation% Description:
% 1. Load sensor IDs and configuration.
% 2. Retrieve best epochs via early stopping logs.
% 3. For each model, apply norm-2 thresholds until reaching 99% of accuracy.
% 4. Evaluate performance in Python, compute F1, show confusion matrices.
% 5. Save final results and plot relevant graphs.
%% =========================%% Initialization and Python environment setup
clc;                % Clear command window
clear;              % Remove all variables from workspace
close all;          % Close all figure windows

% Set up Python environment (use external Python process)
pyenv('Version', 'SELECT DIR...\venv\Scripts\python.EXE', ...
       'ExecutionMode', 'OutOfProcess');

%% Read sensor information from Excel
flowSensID = readmatrix('Info Sensors.xlsx', 'Sheet', 1);   % Flow sensors IDs
pressSensID = readmatrix('Info Sensors.xlsx', 'Sheet', 2);  % Pressure sensors IDs

%% Project settings and model parameters
percAccTuner = 0.99;        % Desired accuracy threshold for tuning
fCost = 1;                  % Cost-sensitive evaluation flag
numero_sensori = 2 + 538 - 2;  % Total number of sensors (flow + pressure - reservoirs)
nSensFlow = 2;              % Number of main flow sensors

pName = 'ProjName';
fMask = 0;                  % Apply mask to main flow sensors (1 = yes, 0 = no)

%% Define paths to weight and info directories
infoDir     = 'INFO/WEIGHTS/';
infoDirW2   = 'INFO/WEIGHTSw2/';
infoDirW3   = 'INFO/WEIGHTSw3/';
infoDirWOut = 'INFO/WEIGHTSwOut/';
infoDirb1   = 'INFO/WEIGHTSb1/';
infoDirb2   = 'INFO/WEIGHTSb2/';
infoDirb3   = 'INFO/WEIGHTSb3/';
infoDirbOut = 'INFO/WEIGHTSbOut/';

infoDirPrec = 'INFO/noNanPrecision/';
infoDirSens = 'INFO/noNanSensitivity/';

saveDir = 'MODELS_WITH_MASK/All/DYN_modelInit0eps_Norm2_FE_a125250_l21opt';

nodeIDDir = 'NODE_ID/dataset.out/';

%% Load grid search parameters and node fault information
gridPar = readmatrix(['INFO\GRID\' pName '\_gridsearch.csv']);
nodeIDfaults = readmatrix([nodeIDDir 'nodeIDTest.csv']);
griglia = gridOp(gridPar);  % Custom function to process grid parameters

%% Initialize weights 
W = [];
weightW2 = [];
weightW3 = [];
weightWOut = [];
weightb1 = [];
weightb2 = [];
weightb3 = [];
weightbOut = [];

%% Load training loss information (for early stopping)
lossInfo = readmatrix('INFO\LOSS\ProjName\log.csv');

% Extract stopping epochs using early stopping strategy
patience = 10; %eARLY STOPPING PARAMETER USED IN PYTH
k = 1;
for i = 2:length(lossInfo)
    if lossInfo(i,1) == 0
        epochs(k,1) = lossInfo(i - 1 - patience,1) + 1;
        epochs(k,2) = lossInfo(i - 1 - patience,4);
        k = k + 1;
    end
end
epochs(k,1) = lossInfo(end - patience,1) + 1;
epochs(k,2) = lossInfo(end - patience,4);

% Update grid search structure with early stopping epoch
for i = 1:length(gridPar)
    gridPar(i,7) = epochs(gridPar(i,4) + 1, 1);
end

%% Main evaluation loop over selected models
for iii = 11:12  % Loop over model indices
    disp(length(gridPar));
    mNum = iii - 1;  % Model index (0-based)

    % Clear weights from previous iteration
    clear W weightW2 weightb1 weightb2 weightbOut weightWOut

    % Load weights for this model
    W          = readmatrix([infoDir     pName '/weight' num2str(mNum) '.csv']);
    weightW2   = readmatrix([infoDirW2   pName '/weight' num2str(mNum) '.csv']);
    weightWOut = readmatrix([infoDirWOut pName '/weight' num2str(mNum) '.csv']);
    weightb1   = readmatrix([infoDirb1   pName '/weight' num2str(mNum) '.csv']);
    weightb2   = readmatrix([infoDirb2   pName '/weight' num2str(mNum) '.csv']);
    weightbOut = readmatrix([infoDirbOut pName '/weight' num2str(mNum) '.csv']);

    % Compute L2 norm thresholds (log-spaced)
    maxVal = max(vecnorm(W', 2));
    minVal = min(vecnorm(W', 2));
    th1 = logspace(log10(minVal), log10(maxVal), 20);

    % Plot L2 norm of weights
    fig = figure();
    fontsize(fig, 14, "points");
    bar(vecnorm(W', 2));
    hold on;

    % Retrieve model accuracy from grid
    ACCURACY_TUNER = gridPar(mNum + 1, 3);

    % Construct save file name
    saveFile = [saveDir pName '/prova' num2str(percAccTuner*100) ...
                num2str(th1(1)) '_' num2str(th1(end)) '_' ...
                num2str(length(th1)) '_' num2str(fCost) '_' ...
                num2str(mNum) '.mat'];

    % Check if the result already exists
    if isfile(saveFile)
        load(saveFile);
    else
        % Begin threshold tuning loop
        thScore = 0;
        j = 1;
        jTh = length(th1);

        while thScore < percAccTuner * ACCURACY_TUNER && jTh > 0
            tic;
            clc;
            disp(j);

            thTested(j) = th1(jTh);  % Current threshold
            p = [];

            % Create binary mask p for weights under threshold
            for i = 1:length(W)
                if vecnorm(W(i,:), 2) + eps < th1(jTh)
                    p(i, 1:width(W)) = 0;
                else
                    p(i, 1:width(W)) = 1;
                end
            end

            pesi_vivi = sum(p, 2);                   % Live weights per input
            tot_pesi_vivi = sum(pesi_vivi);          % Total live weights
            mask1 = double(sum(p, 2) > 0);           % Binary input mask (1 if any weights active)

            weightW1 = p .* W;                       % Masked weights
            segno = sign(W);                         % Sign of original weights
            weightW1(weightW1 == 0) = eps;           % Avoid exact zero values

            % Save mask statistics
            dictMask(:, j) = mask1;
            sensori_di_portata(j) = sum(find(mask1 > 0) <= nSensFlow);
            sensori_di_pressione(j) = sum((find(mask1 > 0) <= numero_sensori) & ...
                                          (find(mask1 > 0) > nSensFlow));
            ora_giorno_attiva(j) = mask1(end);       % Time-of-day feature
            numero_sensori_presenti(j) = sum(mask1);

            % Skip evaluation if same mask as previous iteration
            if j > 1 && numero_sensori_presenti(j) == numero_sensori_presenti(j-1)
                % Copy previous results
                PRED_PROB_TEST{j} = PRED_PROB_TEST{j-1};
                CM_TEST{j} = CM_TEST{j-1};
                Prec_TEST(j) = Prec_TEST(j-1);
                Sens_TEST(j) = Sens_TEST(j-1);
                PRED_PROB_VAL{j} = PRED_PROB_VAL{j-1};
                CM_VAL{j} = CM_VAL{j-1};
                Prec_VAL(j) = Prec_VAL(j-1);
                Sens_VAL(j) = Sens_VAL(j-1);
                LOSS{j} = LOSS{j-1};
                LOSS_VAL{j} = LOSS_VAL{j-1};
                CAT_ACC{j} = CAT_ACC{j-1};
                CAT_ACC_VAL{j} = CAT_ACC_VAL{j-1};
                CAT_ACC_TEST{j} = CAT_ACC_TEST{j-1};
                thScore = CAT_ACC_VAL{j-1}(end);
                AccTuner(j) = ACCURACY_TUNER;
                noNan_Prec_VAL(j) = noNan_Prec_VAL(j-1);
                noNan_Sens_VAL(j) = noNan_Sens_VAL(j-1);
                noNan_Prec_TEST(j) = noNan_Prec_TEST(j-1);
                noNan_Sens_TEST(j) = noNan_Sens_TEST(j-1);
                durata(j) = toc;

                j = j + 1;
                jTh = jTh - 1;
            else
                % Evaluate model using current mask and weights
                if fCost == 1
                    [matTest, avgPrecTest, avgSensTest, predProbTest, ...
                     matVal, avgPrecVal, avgSensVal, predProbVal, ...
                     loss, lossVal, catAcc, catAccVal, catAccTest, ...
                     NaNAvgPrecVal, NaNAvgSensVal, NaNAvgPrecTest, NaNAvgSensTest] = ...
                     pyrunfile("evaluateModelAtDifferentTh.py", ...
                     ["matTest" "avgPrecTest" "avgSensTest" "y_predictedTest" ...
                      "matVal" "avgPrecVal" "avgSensVal" "y_predictedVal" ...
                      "loss" "val_loss" "categorical_accuracy" "val_categorical_accuracy" ...
                      "test_categorical_accuracy" "NaNAvgPrecVal" "NaNAvgSensVal" ...
                      "NaNAvgPrecTest" "NaNAvgSensTest"], ...
                      segno1 = int32(segno), b1 = weightb1, b2 = weightb2, ...
                      bOut = weightbOut, w1 = weightW1, w2 = weightW2, ...
                      wOut = weightWOut, mask = double(mask1), modelNum = int32(mNum), ...
                      projName = pName, flagMask = fMask, maxEpochs = int32(gridPar(iii,7)));
                end

                % Store evaluation results
                PRED_PROB_TEST{j} = double(predProbTest);
                CM_TEST{j} = double(matTest);
                Prec_TEST(j) = avgPrecTest;
                Sens_TEST(j) = avgSensTest;
                noNan_Prec_TEST(j) = NaNAvgPrecTest;
                noNan_Sens_TEST(j) = NaNAvgSensTest;

                PRED_PROB_VAL{j} = double(predProbVal);
                CM_VAL{j} = double(matVal);
                Prec_VAL(j) = avgPrecVal;
                Sens_VAL(j) = avgSensVal;
                noNan_Prec_VAL(j) = NaNAvgPrecVal;
                noNan_Sens_VAL(j) = NaNAvgSensVal;

                LOSS{j} = double(loss);
                LOSS_VAL{j} = double(lossVal);
                CAT_ACC{j} = double(catAcc);
                CAT_ACC_VAL{j} = double(catAccVal);
                CAT_ACC_TEST{j} = double(catAccTest);
                thScore = CAT_ACC_VAL{j}(end);
                AccTuner(j) = ACCURACY_TUNER;
                durata(j) = toc;

                j = j + 1;
                jTh = jTh - 1;
            end
        end

        % Retrieve true one-hot encoded labels via Python
        [trueProb, trueProbVal] = pyrunfile("Fun_retrieveTrueProbs.py", ["dataTestOut" "dataVal2Out"]);
        trueOut = onehotdecode(double(trueProb), [0:7], 2, 'double');
        trueOutVal = onehotdecode(double(trueProbVal), [0:7], 2, 'double');

        % Load pre-threshold metrics
        precPreTh_VAL = readmatrix([infoDirPrec pName '/noNanPrecVal' num2str(mNum) '.csv']);
        sensPreTh_VAL = readmatrix([infoDirSens pName '/noNanSensVal' num2str(mNum) '.csv']);
        precPreTh_TEST = readmatrix([infoDirPrec pName '/noNanPrecTest' num2str(mNum) '.csv']);
        sensPreTh_TEST = readmatrix([infoDirSens pName '/noNanSensTest' num2str(mNum) '.csv']);

        % Save all data
        mkdir([saveDir pName]);
        save(saveFile);

        % Clear all relevant variables for next loop iteration
        noNan_Prec_VAL = [];
        noNan_Sens_VAL = [];
        noNan_Prec_TEST = [];
        noNan_Sens_TEST = [];
        precPreTh_VAL= [];
        sensPreTh_VAL =  [];
        precPreTh_TEST =  [];
        sensPreTh_TEST =  [];
        trueOut=[];
        trueOutVal=[];
        PRED_PROB_TEST = {};
        CM_TEST = {};
        Prec_TEST = [];
        Sens_TEST = []; %"recall/accuracy"
        PRED_PROB_VAL = {};
        CM_VAL = {};
        Prec_VAL = [];
        Sens_VAL = []; %"recall/accuracy"
        LOSS = {};
        LOSS_VAL = {};
        CAT_ACC = {};
        CAT_ACC_VAL = {};
        CAT_ACC_TEST = {};
        AccTuner=[];
        thTested=[];
        dictMask=[];
        sensori_di_portata = [];
        sensori_di_pressione = [];
        ora_giorno_attiva = [];
        numero_sensori_presenti = [];
        W = [];
        weightW2= [];
        weightW3= [];
        weightWOut= [];
        weightb1= [];
        weightb2= [];
        weightb3= [];
        weightbOut= [];
        durata=[];
    end
end
