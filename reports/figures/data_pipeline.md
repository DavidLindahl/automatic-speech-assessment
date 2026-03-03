graph TD
    %% Phase 1: Data Prep
    subgraph P1 [Phase 1: Golden Data Prep]
        A[(Raw Audio & CSVs)] --> B[Caption Generator]
        B -- Sends Meta --> C{{Expert Gemini-2.5}}
        C -- Returns Captions --> D[Golden Dataset<br/>'Chosen' Answers]
    end

    %% Phase 2: SFT
    subgraph P2 [Phase 2: SFT Warmup]
        D --> E[SFT Trainer]
        E -- Mimics Gemini --> F[SFT Warmup Model<br/>'Student V1']
    end

    %% Phase 3: Inference
    subgraph P3 [Phase 3: Generate Mistakes]
        A --> G[Inference Script]
        F --> G
        G -- Model's Own Guesses --> H[Mistakes Data<br/>'Rejected' Answers]
    end

    %% Phase 4: ALLD/DPO
    subgraph P4 [Phase 4: ALLD Aligment]
        D & H --> I[Assembled DPO Triplets]
        
        subgraph Collator [ALLD Data Collator]
            I --> J(Split Paths)
            J --> K[Audio Prompt + Answers]
            J --> L[Text Meta Prompt + Answers]
        end

        K --> M[Policy Model<br/>Audio-LLM Student]
        L --> N[Reference Model<br/>Text-LLM Teacher]

        M & N --> O{DPO Loss}
        O -- Gradient Update --> M
        M --> P[Final Aligned Model]
    end

    %% Styling
    style D fill:#d4edda,stroke:#28a745,stroke-width:2px,color:black
    style F fill:#fff3cd,stroke:#ffc107,stroke-width:2px,color:black
    style H fill:#f8d7da,stroke:#dc3545,stroke-width:2px,color:black
    style N fill:#e2e3e5,stroke:#6c757d,stroke-width:2px,color:black
    style P fill:#cce5ff,stroke:#004085,stroke-width:2px,color:black