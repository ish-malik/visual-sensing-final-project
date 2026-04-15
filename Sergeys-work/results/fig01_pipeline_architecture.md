# F1 Pipeline Architecture

The unified CIS vs DVS pipeline. Ramaa's scene model feeds the scene
event rate and required FPS bridges, those plug into Harshitha's ModuCIS
and Ish's DVS circuit formula by way of the `sensor_database` datasheet
constants, and the two sweeps consume the resulting power functions.

```mermaid
flowchart LR
    subgraph Scene ["Scene model (Ramaa)"]
        R["velocity<br/>obj size<br/>bg density"]
    end

    subgraph Bridges ["Bridges"]
        E["event_rate_at_theta<br/>Ramaa formula scaled 1 over theta"]
        F["cis_required_fps<br/>locked or adaptive"]
    end

    subgraph Specs ["Sensor specs"]
        DB["sensor_database<br/>4 CIS + 4 DVS<br/>datasheet values"]
    end

    subgraph Power ["Power formulas"]
        DP["dvs_power_custom<br/>Ish circuit, theta squared"]
        CP["cis_power_custom<br/>linear interp or ModuCIS LUT"]
    end

    subgraph SweepA ["Sweep A (closed form)"]
        SA["2520 rows<br/>power vs velocity"]
    end

    subgraph SweepB ["Sweep B (tracking)"]
        NM["noise_models<br/>CIS and DVS simulators<br/>coast flag"]
        ST["SORT tracker<br/>HOTA, MOTA, IDF1"]
        SB["MOT17 and synthetic<br/>source rows"]
    end

    subgraph Out ["Outputs"]
        FIG["6 figures<br/>plus design_rule.md"]
    end

    R --> E
    R --> F
    DB --> DP
    DB --> CP
    E --> DP
    F --> CP
    DP --> SA
    CP --> SA
    R --> NM
    DP --> SB
    CP --> SB
    NM --> ST
    ST --> SB
    SA --> FIG
    SB --> FIG
```

The diagram renders in GitHub and VS Code preview. If you need a PNG,
open this file in VS Code and use the Markdown Preview Enhanced export
or paste the code block into `mermaid.live`.
