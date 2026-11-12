function batch_gcsf_embed(inputFolder, outputFolder, params)
% BATCH_GCSF_EMBED  Run gcsf_embed over every <id>_nodes.csv / <id>_edges.csv
% pair in inputFolder, writing <id>_graphPE.csv (and optionally
% <id>_nodePE.csv, <id>_syncPE.csv) to outputFolder.
%
%   batch_gcsf_embed('data/in', 'data/out', params)
%
% params is passed to gcsf_embed for every graph; set params.write_node_csv
% to true to also write node encodings, and params.index_base to 0 or 1 if
% any edge file has ambiguous indexing.
    if nargin < 3, params = struct(); end
    if ~exist(outputFolder, "dir"), mkdir(outputFolder); end
    if ~isfield(params,'write_node_csv'), params.write_node_csv = false; end
    nodeFiles = dir(fullfile(inputFolder, '*_nodes.csv'));
    if isempty(nodeFiles)
        warning('No *_nodes.csv files found in: %s', inputFolder);
        return;
    end
    nTotal = numel(nodeFiles); nDone = 0; nSkip = 0; nFail = 0;
    for ii = 1:nTotal
        nodeFileName = nodeFiles(ii).name;
        graphID      = extractBefore(nodeFileName, '_nodes.csv');
        nodePath     = fullfile(inputFolder, [graphID '_nodes.csv']);
        edgePath     = fullfile(inputFolder, [graphID '_edges.csv']);
        if ~exist(edgePath, "file")
            warning('Skipping %s (missing edge file)', graphID);
            nSkip = nSkip + 1; continue;
        end
        params2 = params;
        if params2.write_node_csv
            outNodeCSV = fullfile(outputFolder, [graphID '_nodePE.csv']);
        else
            outNodeCSV = "";
        end
        if isfield(params2,'compute_sync') && params2.compute_sync
            params2.outSyncCSV = fullfile(outputFolder, [graphID '_syncPE.csv']);
        else
            params2.outSyncCSV = "";
        end
        params2.outGraphCSV = fullfile(outputFolder, [graphID '_graphPE.csv']);
        fprintf('[%d/%d] GCSF embedding: %s\n', ii, nTotal, graphID);
        try
            gcsf_embed(nodePath, edgePath, outNodeCSV, params2);
            nDone = nDone + 1;
        catch ME
            warning('FAILED on %s: %s', graphID, ME.message);
            nFail = nFail + 1;
        end
    end
    fprintf('Batch complete. Done=%d, Skipped=%d, Failed=%d (Total=%d)\n', ...
        nDone, nSkip, nFail, nTotal);
end
