%% =========================
% MATLAB SCRIPT: Single Threshold Evaluation with Pretrained Weights
% Description:
% 1. Load sensor IDs and configuration.
% 2. Retrieve best epochs via early stopping logs.
% 3. For each model, apply a fixed norm-2 threshold to prune inputs.
% 4. Evaluate performance in Python, compute F1, show confusion matrices.
% 5. Save final results and plot relevant graphs.
%% =========================%% Initialization and Python environment setup
clc;                % Clear command window
clear;              % Remove all variables from workspace
close all;          % Close all figure windows

% Set up Python environment (use external Python process)
pyenv('Version', 'SELECT DIR...\venv\Scripts\python.EXE', ...
       'ExecutionMode', 'OutOfProcess');

%% === 1. Load Sensor IDs and Basic Configuration ===
flowSensID = readmatrix('Info Sensors.xlsx','Sheet',1);  % Flow sensor IDs 
pressSensID = readmatrix('Info Sensors.xlsx','Sheet',2); % Pressure sensor IDs

percAccTuner = 0.99;                % Target accuracy threshold
fCost = 1;                          % Cost-sensitive flag (1=use cost function)
numero_sensori = 2 + 538 - 2;       % Total sensors: flow + pressure - 2 reservoir
nSensFlow = 2;                      % Number of main flow sensors

pName = 'ProjName';
fMask = 0;                          % 1 = insert mask for the 4 main flow sensors

% Directory paths for weights
infoDir     = 'INFO/WEIGHTS/';
infoDirW2   = 'INFO/WEIGHTSw2/';
infoDirW3   = 'INFO/WEIGHTSw3/';
infoDirWOut = 'INFO/WEIGHTSwOut/';
infoDirb1   = 'INFO/WEIGHTSb1/';
infoDirb2   = 'INFO/WEIGHTSb2/';
infoDirb3   = 'INFO/WEIGHTSb3/';
infoDirbOut = 'INFO/WEIGHTSbOut/';

% Directories for precision/sensitivity logs and final saving
infoDirPrec = 'INFO/noNanPrecision/';
infoDirSens = 'INFO/noNanSensitivity/';
saveDir     = 'MODELS_WITH_MASK/All/ProjName';

% Directory for dynamic threshold loading
loadDirDynTh = 'MODELS_WITH_MASK/All/ProjName';

% Paths for node IDs and grid search info
nodeIDDir = 'NODE_ID/dataset.out/';
gridPar = readmatrix(['INFO\GRID\' pName '\_gridsearch.csv']);
nodeIDfaults = readmatrix([nodeIDDir 'nodeIDTest.csv']);  % Node IDs used for test
griglia = gridOp(gridPar); % Some user-defined function for grid operations

%% === 2. Initialize and Prepare Weight Variables ===
W = [];
weightW2 = [];
weightW3 = [];
weightWOut = [];
weightb1 = [];
weightb2 = [];
weightb3 = [];
weightbOut = [];

% Load training log (lossInfo) to retrieve best epochs from early stopping
lossInfo = readmatrix('..\ProjName\log.csv');

%% === 3. Extract Early Stopping Epochs ===
patience = 10;  % Early stopping patience
k = 1;
for i = 2:length(lossInfo)
    if lossInfo(i,1) == 0
        epochs(k,1) = lossInfo(i-1-patience,1) + 1;
        epochs(k,2) = lossInfo(i-1-patience,4);
        k = k + 1;
    end
end
epochs(k,1) = lossInfo(end - patience,1) + 1;
epochs(k,2) = lossInfo(end - patience,4);

% Update gridPar with the best epochs found
for i = 1:length(gridPar)
    gridPar(i,7) = epochs(gridPar(i,4) + 1, 1);
end

%% === 4. Main Loop Over All Models ===
for iii = 1:14
    mNum = iii - 1;  % Model index (0 = best model, up to numOfTrials - 1)

    % Clear previous iteration's weights
    clear W weightW2 weightWOut weightb1 weightb2 weightbOut

    % Load model weights
    W          = readmatrix([infoDir     pName '/weight' num2str(mNum) '.csv']);
    weightW2   = readmatrix([infoDirW2   pName '/weight' num2str(mNum) '.csv']);
    weightWOut = readmatrix([infoDirWOut pName '/weight' num2str(mNum) '.csv']);
    weightb1   = readmatrix([infoDirb1   pName '/weight' num2str(mNum) '.csv']);
    weightb2   = readmatrix([infoDirb2   pName '/weight' num2str(mNum) '.csv']);
    weightbOut = readmatrix([infoDirbOut pName '/weight' num2str(mNum) '.csv']);

    % Extract reference accuracy from grid
    ACCURACY_TUNER = gridPar(mNum+1,3);

    % === 4.1 Choose a single threshold (example: norm-2 ~ 0.759)
    th1 = 0.759156578984714;

    % === 4.2 Check if result is already saved
    resultPath = [saveDir pName '/prova' num2str(percAccTuner*100) '_' num2str(length(th1)) '_' num2str(fCost) '_' num2str(mNum) '.mat'];
    if isfile(resultPath)
        % If results exist, just load them
        load(resultPath);
    else
        % Otherwise, evaluate the model at the chosen threshold

        thScore = 0;
        j = 1;
        jTh = length(th1);

        clc; disp(j);

        % Store threshold tested
        thTested(j) = th1(jTh);

        % Create the pruning mask p (norm-2 of each row vs threshold)
        p = [];
        for i = 1:size(W,1)
            if vecnorm(W(i,:),2) + eps < th1(jTh)
                p(i,1:width(W)) = 0;
            else
                p(i,1:width(W)) = 1;
            end
        end

        % Calculate how many weights survived
        pesi_vivi = sum(p');
        tot_pesi_vivi = sum(pesi_vivi);

        % Create final input mask (mask1) to remove sensors w/ no active weights
        mask1 = zeros(size(p,1),1);
        for i = 1:length(p)
            if sum(p(i,:)) == 0
                mask1(i,1) = 0;
            else
                mask1(i,1) = 1;
            end
        end

        % Apply mask to weights
        weightW1 = p .* W;
        segno = sign(W);
        weightW1(weightW1 == 0) = eps;

        dictMask(:,j) = mask1;

        % Count how many flow and pressure sensors remain
        sensori_di_portata(j) = length(find(find(mask1>0) <= nSensFlow));
        sensori_di_pressione(j) = length(find(find(mask1>0) <= numero_sensori & find(mask1>0) > nSensFlow));
        ora_giorno_attiva(j) = mask1(end);
        numero_sensori_presenti(j) = length(mask1) - sum(mask1 == 0);

        % === 4.3 Avoid re-evaluating if sensor set didn't change
        if j > 1 && numero_sensori_presenti(j) == numero_sensori_presenti(j-1)
            % Copy previous iteration's results if mask is identical
            PRED_PROB_TEST{j}  = PRED_PROB_TEST{j-1};
            CM_TEST{j}         = CM_TEST{j-1};
            Prec_TEST(j)       = Prec_TEST(j-1);
            Sens_TEST(j)       = Sens_TEST(j-1);
            PRED_PROB_VAL{j}   = PRED_PROB_VAL{j-1};
            CM_VAL{j}          = CM_VAL{j-1};
            Prec_VAL(j)        = Prec_VAL(j-1);
            Sens_VAL(j)        = Sens_VAL(j-1);
            LOSS{j}            = LOSS{j-1};
            LOSS_VAL{j}        = LOSS_VAL{j-1};
            CAT_ACC{j}         = CAT_ACC{j-1};
            CAT_ACC_VAL{j}     = CAT_ACC_VAL{j-1};
            CAT_ACC_TEST{j}    = CAT_ACC_TEST{j-1};
            thScore            = CAT_ACC_VAL{j-1}(end);
            AccTuner(j)        = ACCURACY_TUNER;
            noNan_Prec_VAL(j)  = noNan_Prec_VAL(j-1);
            noNan_Sens_VAL(j)  = noNan_Sens_VAL(j-1);
            noNan_Prec_TEST(j) = noNan_Prec_TEST(j-1);
            noNan_Sens_TEST(j) = noNan_Sens_TEST(j-1);

            j   = j + 1;
            jTh = jTh - 1;

        else
            % === 4.4 Evaluate model in Python if set changed
            if fCost == 1
                tic;
                [matTest, avgPrecTest, avgSensTest, predProbTest, ...
                 matVal, avgPrecVal, avgSensVal, predProbVal, ...
                 loss, lossVal, catAcc, catAccVal, catAccTest, ...
                 NaNAvgPrecVal, NaNAvgSensVal, NaNAvgPrecTest, NaNAvgSensTest] = ...
                    pyrunfile("evaluateModelAtSingleTh.py", ...
                     ["matTest" "avgPrecTest" "avgSensTest" "y_predictedTest" "matVal" "avgPrecVal" "avgSensVal" "y_predictedVal" "loss" "val_loss" "categorical_accuracy" "val_categorical_accuracy" "test_categorical_accuracy" "NaNAvgPrecVal" "NaNAvgSensVal" "NaNAvgPrecTest" "NaNAvgSensTest"], ...
                     segno1=int32(segno), b1=weightb1, b2=weightb2, bOut=weightbOut, ...
                     w1=weightW1, w2=weightW2, wOut=weightWOut, mask=double(mask1), ...
                     modelNum=int32(mNum), projName=pName, flagMask=fMask, ...
                     maxEpochs=int32(gridPar(iii,7)));
                durata(iii) = toc;
            end

            % Store results
            PRED_PROB_TEST{j}   = double(predProbTest);
            CM_TEST{j}          = double(matTest);
            Prec_TEST(j)        = avgPrecTest;
            Sens_TEST(j)        = avgSensTest;
            noNan_Prec_TEST(j)  = NaNAvgPrecTest;
            noNan_Sens_TEST(j)  = NaNAvgSensTest;
            PRED_PROB_VAL{j}    = double(predProbVal);
            CM_VAL{j}           = double(matVal);
            Prec_VAL(j)         = avgPrecVal;
            Sens_VAL(j)         = avgSensVal;
            noNan_Prec_VAL(j)   = NaNAvgPrecVal;
            noNan_Sens_VAL(j)   = NaNAvgSensVal;
            LOSS{j}             = double(loss);
            LOSS_VAL{j}         = double(lossVal);
            CAT_ACC{j}          = double(catAcc);
            CAT_ACC_VAL{j}      = double(catAccVal);
            CAT_ACC_TEST{j}     = double(catAccTest);
            thScore             = CAT_ACC_VAL{j}(end);
            AccTuner(j)         = ACCURACY_TUNER;

            j = j + 1;
            jTh = jTh - 1;
        end
    end

    % === 4.5 Load ground truth after evaluation
    [trueProb, trueProbVal] = pyrunfile("Fun_retrieveTrueProbs.py", ["dataTestOut", "dataVal2Out"]);
    trueOut    = onehotdecode(double(trueProb),    [0 1 2 3 4 5 6 7], 2, 'double');
    trueOutVal = onehotdecode(double(trueProbVal), [0 1 2 3 4 5 6 7], 2, 'double');

    % Load data from dynamic threshold results for reference
    precSensTestDynTh = load([ loadDirDynTh '/prova99_20_1_' num2str(mNum) '.mat'], ...
                             'CM_TEST','PRED_PROB_TEST','noNan_Prec_TEST','noNan_Sens_TEST');

    % Read pre-threshold precision/sensitivity
    precPreTh_VAL = readmatrix([infoDirPrec pName '/noNanPrecVal' num2str(mNum) '.csv']);
    sensPreTh_VAL = readmatrix([infoDirSens pName '/noNanSensVal' num2str(mNum) '.csv']);
    precPreTh_TEST = readmatrix([infoDirPrec pName '/noNanPrecTest' num2str(mNum) '.csv']);
    sensPreTh_TEST = readmatrix([infoDirSens pName '/noNanSensTest' num2str(mNum) '.csv']);

    % If not saved yet, create directory and save results
    if ~isfile(resultPath)
        mkdir([saveDir pName]);
        save(resultPath);
    end

    % Clear large variables before next iteration
    precSensTestDynTh = [];
    noNan_Prec_VAL = []; noNan_Sens_VAL = [];
    noNan_Prec_TEST = []; noNan_Sens_TEST = [];
    precPreTh_VAL = []; sensPreTh_VAL = [];
    precPreTh_TEST = []; sensPreTh_TEST = [];
    trueOut = []; trueOutVal = [];
    PRED_PROB_TEST = {}; CM_TEST = {};
    Prec_TEST = []; Sens_TEST = [];
    PRED_PROB_VAL = {}; CM_VAL = {};
    Prec_VAL = []; Sens_VAL = [];
    LOSS = {}; LOSS_VAL = {};
    CAT_ACC = {}; CAT_ACC_VAL = {};
    CAT_ACC_TEST = {};
    AccTuner = []; thTested = []; dictMask = [];
    sensori_di_portata = []; sensori_di_pressione = [];
    ora_giorno_attiva = []; numero_sensori_presenti = [];
    W = []; weightW2 = []; weightW3 = []; weightWOut = [];
    weightb1 = []; weightb2 = []; weightb3 = []; weightbOut = [];
end

%% === 5. Compute F1 Scores and Plot ===
f1    = 2 * (Prec_TEST .* Sens_TEST)     ./ (Prec_TEST + Sens_TEST);
f1Val = 2 * (Prec_VAL  .* Sens_VAL)      ./ (Prec_VAL  + Sens_VAL);

% Plot results for Test
accuracygraph(mNum, thTested, Sens, Prec, f1, numero_sensori_presenti);

catAccuracy = zeros(length(thTested),1);
for j = 1:length(thTested)
    catAccuracy(j,1) = CAT_ACC_VAL{j}(end);
end

% Plot results for Validation
accuracygraph(mNum, thTested, SensVal, PrecVal, f1Val, ...
              numero_sensori_presenti, gridPar, catAccuracy);

%% === 6. Graph Probability & Confusion Matrix ===
maxf1ind = find(f1 == max(f1));
nNorm = numero_sensori_presenti / max(numero_sensori_presenti);
provaScore = f1 + 1 ./ (1 + nNorm);

j = maxf1ind(end);

figure();
plot(f1,'r'); legend('f1');

GraphProb(PRED_PROB{j}, trueOut, nodeIDfaults, flowSensID(:,1), pressSensID(:,1));

% Confusion Matrices
ConfMatrix(CM{j},       trueOut, 'Test');
ConfMatrix(CM_VAL{j},   trueOut, 'Val');

% Display sensor info
disp(['Sensors present:' num2str(numero_sensori_presenti(j))]);
disp(['Pressure sensors:' num2str(sensori_di_pressione(j))]);
disp(['Flow sensors:' num2str(sensori_di_portata(j))]);

[flowID, pressID] = findsensorid(dictMask(:,j), nSensFlow, ...
                                 flowSensID(:,1), pressSensID(:,1));
