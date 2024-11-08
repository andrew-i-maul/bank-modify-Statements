flowchart TB
    Start([Start]) --> LoadRules[Load Mapping Rules]
    LoadRules --> CreateOutDir[Create Output Directory]
    CreateOutDir --> InitStats[Initialize Statistics]
    
    subgraph ProcessDirectory[Process Directory]
        InitStats --> NextFile{More CSV Files?}
        NextFile -->|Yes| ProcessFile[Process Bank Statement]
        ProcessFile --> UpdateStats[Update Statistics]
        UpdateStats --> NextFile
        NextFile -->|No| SaveRules[Save Updated Rules]
    end
    
    subgraph ProcessBankStatement[Process Bank Statement]
        ProcessFile --> ReadCSV[Read CSV File]
        ReadCSV --> ExtractInfo[Extract Bank/Owner Info]
        ExtractInfo --> ProcessRows[Process Each Row]
    end
    
    subgraph ProcessRow[Row Processing]
        <!-- ProcessRows --> CleanDesc[Clean Description]v -->
        CleanDesc --> IsChanged{Already Mapped?}
        IsChanged -->|No| ApplyRules[Apply Mapping Rules]
        IsChanged -->|Yes| SkipRow[Skip Row]
        ApplyRules --> MatchFound{Pattern Match?}
        MatchFound -->|Yes| UpdateRow[Update Row Values]
        MatchFound -->|No| NoChange[No Changes Made]
    end
    
    SaveRules --> LogStats[Log Statistics]
    LogStats --> End([End])

    %% Styling
    classDef process fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef decision fill:#fff3e0,stroke:#ef6c00,stroke-width:2px;
    classDef terminal fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef subgraph fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px;
    
    class Start,End terminal;
    class LoadRules,CreateOutDir,InitStats,ProcessFile,ReadCSV,ExtractInfo,ProcessRows,CleanDesc,ApplyRules,UpdateRow,NoChange,SaveRules,LogStats,UpdateStats,SkipRow process;
    <!-- class NextFile,vIsChanged,MatchFound decision; -->
    class ProcessDirectory,ProcessBankStatement,ProcessRow subgraph;