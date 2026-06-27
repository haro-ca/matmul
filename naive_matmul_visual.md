# Naive Matmul - Visual Explanation

## Thread Mapping for 4×4 Matrices

```mermaid
graph TB
    subgraph "GPU Grid: 2×2 blocks"
        direction TB
        subgraph "Block (0,0)"
            B0_T0["Thread (0,0)<br/>→ C[0,0]"]
            B0_T1["Thread (1,0)<br/>→ C[0,1]"]
            B0_T2["Thread (0,1)<br/>→ C[1,0]"]
            B0_T3["Thread (1,1)<br/>→ C[1,1]"]
            B0_T0 --- B0_T1
            B0_T2 --- B0_T3
            B0_T0 --- B0_T2
            B0_T1 --- B0_T3
        end
        
        subgraph "Block (1,0)"
            B1_T0["Thread (0,0)<br/>→ C[0,2]"]
            B1_T1["Thread (1,0)<br/>→ C[0,3]"]
            B1_T2["Thread (0,1)<br/>→ C[1,2]"]
            B1_T3["Thread (1,1)<br/>→ C[1,3]"]
            B1_T0 --- B1_T1
            B1_T2 --- B1_T3
            B1_T0 --- B1_T2
            B1_T1 --- B1_T3
        end
        
        subgraph "Block (0,1)"
            B2_T0["Thread (0,0)<br/>→ C[2,0]"]
            B2_T1["Thread (1,0)<br/>→ C[2,1]"]
            B2_T2["Thread (0,1)<br/>→ C[3,0]"]
            B2_T3["Thread (1,1)<br/>→ C[3,1]"]
            B2_T0 --- B2_T1
            B2_T2 --- B2_T3
            B2_T0 --- B2_T2
            B2_T1 --- B2_T3
        end
        
        subgraph "Block (1,1)"
            B3_T0["Thread (0,0)<br/>→ C[2,2]"]
            B3_T1["Thread (1,0)<br/>→ C[2,3]"]
            B3_T2["Thread (0,1)<br/>→ C[3,2]"]
            B3_T3["Thread (1,1)<br/>→ C[3,3]"]
            B3_T0 --- B3_T1
            B3_T2 --- B3_T3
            B3_T0 --- B3_T2
            B3_T1 --- B3_T3
        end
    end
```

## Thread C[0,0] Execution Flow

```mermaid
sequenceDiagram
    participant T as Thread (0,0)
    participant GM as Global Memory
    participant C as Output C[0,0]
    
    T->>T: row, col = cuda.grid(2)<br/>row=0, col=0
    T->>T: if 0<4 and 0<4 → True
    T->>T: acc = 0.0
    
    Note over T,GM: Loop k=0 to 3
    T->>GM: Read A[0,0]
    GM-->>T: a00
    T->>GM: Read B[0,0]
    GM-->>T: b00
    T->>T: acc += a00 * b00
    
    T->>GM: Read A[0,1]
    GM-->>T: a01
    T->>GM: Read B[1,0]
    GM-->>T: b10
    T->>T: acc += a01 * b10
    
    T->>GM: Read A[0,2]
    GM-->>T: a02
    T->>GM: Read B[2,0]
    GM-->>T: b20
    T->>T: acc += a02 * b20
    
    T->>GM: Read A[0,3]
    GM-->>T: a03
    T->>GM: Read B[3,0]
    GM-->>T: b30
    T->>T: acc += a03 * b30
    
    T->>C: C[0,0] = acc
```

## Memory Access Pattern - The Problem

```mermaid
graph LR
    subgraph "Thread (0,0) computes C[0,0]"
        T0["Thread (0,0)"]
        T0 --> |reads| A0["A[0,0], A[0,1], A[0,2], A[0,3]"]
        T0 --> |reads| B0["B[0,0], B[1,0], B[2,0], B[3,0]"]
    end
    
    subgraph "Thread (1,0) computes C[0,1]"
        T1["Thread (1,0)"]
        T1 --> |reads| A1["A[0,0], A[0,1], A[0,2], A[0,3]"]
        T1 --> |reads| B1["B[0,1], B[1,1], B[2,1], B[3,1]"]
    end
    
    subgraph "Thread (0,1) computes C[1,0]"
        T2["Thread (0,1)"]
        T2 --> |reads| A2["A[1,0], A[1,1], A[1,2], A[1,3]"]
        T2 --> |reads| B2["B[0,0], B[1,0], B[2,0], B[3,0]"]
    end
    
    A0 -.->|DUPLICATE| A1
    B0 -.->|DUPLICATE| B2
    
    style A0 fill:#ff9999
    style A1 fill:#ff9999
    style B0 fill:#ff9999
    style B2 fill:#ff9999
```

## Data Redundancy Problem

```mermaid
graph TB
    subgraph "4×4 Matrix A (16 elements)"
        A["a00 a01 a02 a03<br/>a10 a11 a12 a13<br/>a20 a21 a22 a23<br/>a30 a31 a32 a33"]
    end
    
    subgraph "Thread Access Patterns"
        T0["Thread C[0,0]:<br/>reads row 0"]
        T1["Thread C[0,1]:<br/>reads row 0"]
        T2["Thread C[0,2]:<br/>reads row 0"]
        T3["Thread C[0,3]:<br/>reads row 0"]
    end
    
    A --> T0
    A --> T1
    A --> T2
    A --> T3
    
    R["Row 0 read 4 times!"]
    T0 --> R
    T1 --> R
    T2 --> R
    T3 --> R
    
    style R fill:#ff6b6b,color:#fff
```

## Why Naive is Memory-Bound

```mermaid
pie title "Time Breakdown per Thread"
    "Waiting for memory" : 95
    "Actual arithmetic" : 5
```

```mermaid
graph LR
    subgraph "Per Thread Operation"
        M1["Memory read A[row,k]<br/>~400 cycles"]
        M2["Memory read B[k,col]<br/>~400 cycles"]
        COMP["Multiply + Add<br/>~1 cycle"]
    end
    
    M1 --> COMP
    M2 --> COMP
    
    style M1 fill:#ff9999
    style M2 fill:#ff9999
    style COMP fill:#99ff99
```

## Scaling to 1024×1024

```mermaid
graph TB
    subgraph "1024×1024 Matmul"
        T["Total threads: 1,048,576"]
        R["Reads per thread: 2,048"]
        TR["Total reads: 2.1 billion"]
        U["Unique elements: 2,048"]
    end
    
    T --> R
    R --> TR
    U --> R
    
    WASTED["Wasted reads: ~1000x redundancy"]
    TR --> WASTED
    
    style WASTED fill:#ff6b6b,color:#fff
```

## How to Import to Excalidraw

1. Copy the Mermaid code blocks above
2. Go to [excalidraw.com](https://excalidraw.com)
3. Enable Mermaid support (Menu → Enable Mermaid)
4. Paste the code into a text element or use the Mermaid library
5. The diagrams will render as editable vector graphics

Alternatively, you can:
- Use [Mermaid Live Editor](https://mermaid.live) to preview and export as SVG/PNG
- Copy the SVG/PNG into Excalidraw for further annotation
- Use Excalidraw's built-in shape library to recreate the concepts manually