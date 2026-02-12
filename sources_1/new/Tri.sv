//This is simulation only example for training of 4 layer Kolmogorov-Arnold networks to predict areas of 
//random triangles given by coordinates of the vertices
//Concept of Mike Poluektov and Andrew Polar. 
//Coding by Andrew Polar. 
//Publications: https://www.sciencedirect.com/science/article/abs/pii/S0952197620303742
//https://link.springer.com/article/10.1007/s10994-025-06800-6
//All computations in integers, no divisions, code fit massive board parallel processing. 
//The end result is relative error for validation set 18290. The target limit is 1805000, so it is near 1%.
//When testing on board LEDs may be set to show the either error or number of cycles. 
//When designed professionally the latency for one record must be 20 to 30 cycles and this latency not 
//depend on width of the model, only on the number of layers. So if board hardware allows your  4 layer model
//runs with the same processing speed. 
module Tri #(
    //data
    parameter int N_FEATURES   = 6,
    parameter int N_T_RECORDS  = 8192,
    parameter int N_V_RECORDS  = 2048,
    parameter int N_EPOCHS     = 32,
    
    //network, blocks in layers
    parameter int N_U0         = 53,
    parameter int N_U1         = 11,
    parameter int N_U2         = 4,
    parameter int N_U3         = 1,
    
    //common 
    parameter int N_BASE_SHIFT = 13,
    
    //layer0
    parameter int N_POINTS0      = 2,
    parameter int N_XMIN0        = 0,
    parameter int N_XMAX0        = 2048,
    parameter int N_DELTA_SHIFT0 = 11,
    parameter int N_ALPHA_SHIFT0 = 13,
    parameter int N_MULT0        = 1365,
    
    //layer1 
    parameter int N_POINTS1      = 14,
    parameter int N_XMIN1        = -10000,
    parameter int N_XMAX1        = 1693936,
    parameter int N_DELTA_SHIFT1 = 17,
    parameter int N_ALPHA_SHIFT1 = 8,
    parameter int N_MULT1        = 154,    
    
     //layer2 
    parameter int N_POINTS2      = 14,
    parameter int N_XMIN2        = -10000,
    parameter int N_XMAX2        = 1693936,
    parameter int N_DELTA_SHIFT2 = 17,
    parameter int N_ALPHA_SHIFT2 = 7,
    parameter int N_MULT2        = 744,       
    
     //layer3 
    parameter int N_POINTS3      = 14,
    parameter int N_XMIN3        = -10000,
    parameter int N_XMAX3        = 1693936,
    parameter int N_DELTA_SHIFT3 = 17,
    parameter int N_ALPHA_SHIFT3 = 6,
    parameter int N_MULT3        = 2048       
)( 
    input  logic        CLK100MHZ,
    input  logic        CPU_RESETN,
    output logic [15:0] LED
);

    //data
    logic signed [31:0] flat_features_training   [0:N_T_RECORDS*N_FEATURES-1];
    logic signed [31:0] flat_features_validation [0:N_V_RECORDS*N_FEATURES-1];
    logic signed [31:0] targets_training         [0:N_T_RECORDS - 1];
    logic signed [31:0] targets_validation       [0:N_V_RECORDS - 1];
    
    //models
    logic signed [31:0] flat_level0 [0:N_FEATURES*N_U0*N_POINTS0-1];
    logic signed [31:0] flat_level1 [0:N_U0*N_U1*N_POINTS1-1];
    logic signed [31:0] flat_level2 [0:N_U1*N_U2*N_POINTS2-1];
    logic signed [31:0] flat_level3 [0:N_U2*N_U3*N_POINTS3-1];
    
    //auxiliary buffers 
    logic signed [31:0] models0 [0:N_U0-1];
    logic signed [31:0] models1 [0:N_U1-1];
    logic signed [31:0] models2 [0:N_U2-1];
    logic signed [31:0] models3 [0:N_U3-1];
    
    logic signed [63:0] deltas3 [0:N_U3-1]; 
    logic signed [63:0] deltas2 [0:N_U2-1]; 
    logic signed [63:0] deltas1 [0:N_U1-1]; 
    logic signed [63:0] deltas0 [0:N_U0-1]; 
    
    logic signed [31:0] differences2[0:N_U3-1][0:N_U2-1];
    logic signed [31:0] differences1[0:N_U2-1][0:N_U1-1];
    logic signed [31:0] differences0[0:N_U1-1][0:N_U0-1];

    logic signed [31:0] buffer[0:N_U0-1][0:N_U0-1];
    
    //function structures
    typedef struct {
        logic signed [31:0] f [0:15];  //it is simply max possible size
        logic signed [31:0] xmin;
        logic signed [31:0] xmax;
        logic signed [31:0] offset;
        logic [5:0] index; 
        logic [5:0] delta_shift;  
        int          nPoints;          
    } Function_t;          
    Function_t U0 [0:N_FEATURES*N_U0-1];
    Function_t U1 [0:N_U0*N_U1-1];
    Function_t U2 [0:N_U1*N_U2-1];
    Function_t U3 [0:N_U2*N_U3-1];
    
    int error;
int record, k, j;             // loop indices
logic signed [31:0] e;        // temporary error per record
logic signed [63:0] m0, m1, m2, m3; // accumulation variables


initial begin
        // Load data
        flat_features_training   = '{`include "features_training.svh"};
        flat_features_validation = '{`include "features_validation.svh"};
        targets_training         = '{`include "targets_training.svh"};
        targets_validation       = '{`include "targets_validation.svh"};
        
        //load raw initial model data
        flat_level0 = '{`include "layer_zero.svh"};
        flat_level1 = '{`include "layer_one.svh"};
        flat_level2 = '{`include "layer_two.svh"};
        flat_level3 = '{`include "layer_three.svh"};
        
        //initialize functions
        for (int i = 0, k = 0; i < N_FEATURES * N_U0; i++) begin
            Initialize(U0[i], flat_level0, N_XMIN0, N_XMAX0, N_DELTA_SHIFT0, N_POINTS0, k);
            k = k + N_POINTS0;
        end
        //
        for (int i = 0, k = 0; i < N_U0 * N_U1; i++) begin
            Initialize(U1[i], flat_level1, N_XMIN1, N_XMAX1, N_DELTA_SHIFT1, N_POINTS1, k);
            k = k + N_POINTS1;
        end
        //
        for (int i = 0, k = 0; i < N_U1 * N_U2; i++) begin
            Initialize(U2[i], flat_level2, N_XMIN2, N_XMAX2, N_DELTA_SHIFT2, N_POINTS2, k);
            k = k + N_POINTS2;
        end
        //
        for (int i = 0, k = 0; i < N_U2 * N_U3; i++) begin
            Initialize(U3[i], flat_level3, N_XMIN3, N_XMAX3, N_DELTA_SHIFT3, N_POINTS3, k);
            k = k + N_POINTS3;
        end     
        
        //from here to the end of initial block is training
        for (int epoch = 0; epoch < N_EPOCHS; ++epoch) begin
            for (int record = 0; record < N_T_RECORDS; ++record) begin
            
                // =====================
                // Forward pass
                // =====================
                
                // Block #1: layer 0 compute
                for (int k = 0; k < N_U0; ++k) begin
                    for (int j = 0; j < N_FEATURES; ++j) begin
                        buffer[k][j] = Compute(training_feature_at(record, j), U0[k * N_FEATURES + j]);
                    end
                end
                
                for (int k = 0; k < N_U0; ++k) begin
                    models0[k] = reduce_row(k, N_FEATURES, N_MULT0);
                end

                // Block #3: layer 1 compute
                for (int k = 0; k < N_U1; ++k) begin
                    for (int j = 0; j < N_U0; ++j) begin
                        buffer[k][j] = Compute(models0[j], U1[k * N_U0 + j]);
                    end
                end
                
                for (int k = 0; k < N_U1; ++k) begin
                    models1[k] = reduce_row(k, N_U0, N_MULT1);
                end

                // Block #5: layer 2 compute
                for (int k = 0; k < N_U2; ++k) begin
                    for (int j = 0; j < N_U1; ++j) begin
                        buffer[k][j] = Compute(models1[j], U2[k * N_U1 + j]);
                    end
                end
                
                for (int k = 0; k < N_U2; ++k) begin
                    models2[k] = reduce_row(k, N_U1, N_MULT2);
                end

                // Block #7: layer 3 compute
                for (int k = 0; k < N_U3; ++k) begin
                    for (int j = 0; j < N_U2; ++j) begin
                        buffer[k][j] = Compute(models2[j], U3[k * N_U2 + j]);
                    end
                end
                
                for (int k = 0; k < N_U3; ++k) begin
                    models3[k] = reduce_row(k, N_U2, N_MULT3);
                end

                // =====================
                // Backward pass
                // =====================

                // Block #9: differences
                for (int k = 0; k < N_U3; ++k)
                    for (int j = 0; j < N_U2; ++j)
                        differences2[k][j] = GetDifference(U3[k * N_U2 + j]);

                for (int k = 0; k < N_U2; ++k)
                    for (int j = 0; j < N_U1; ++j)
                        differences1[k][j] = GetDifference(U2[k * N_U1 + j]);

                for (int k = 0; k < N_U1; ++k)
                    for (int j = 0; j < N_U0; ++j)
                        differences0[k][j] = GetDifference(U1[k * N_U0 + j]);

                // Block #10: deltas3 -> deltas2
                deltas3[0] = $signed(targets_training[record]) - models3[0];

                for (int j = 0; j < N_U2; ++j) begin
                    deltas2[j] = 0;
                    for (int i = 0; i < N_U3; ++i)
                        deltas2[j] += ($signed(differences2[i][j]) * deltas3[i]) >>> N_DELTA_SHIFT3;
                end

                // Block #11: deltas1
                for (int j = 0; j < N_U1; ++j) begin
                    deltas1[j] = 0;
                    for (int i = 0; i < N_U2; ++i)
                        deltas1[j] += ($signed(differences1[i][j]) * deltas2[i]) >>> N_DELTA_SHIFT2;
                end

                // Block #12: deltas0
                for (int j = 0; j < N_U0; ++j) begin
                    deltas0[j] = 0;
                    for (int i = 0; i < N_U1; ++i)
                        deltas0[j] += ($signed(differences0[i][j]) * deltas1[i]) >>> N_DELTA_SHIFT1;
                end

                // Block #13: updates
                for (int k = 0; k < N_U3; ++k)
                    for (int j = 0; j < N_U2; ++j)
                        Update(deltas3[k] >>> N_ALPHA_SHIFT3, U3[k * N_U2 + j]);

                for (int k = 0; k < N_U2; ++k)
                    for (int j = 0; j < N_U1; ++j)
                        Update(deltas2[k] >>> N_ALPHA_SHIFT2, U2[k * N_U1 + j]);

                for (int k = 0; k < N_U1; ++k)
                    for (int j = 0; j < N_U0; ++j)
                        Update(deltas1[k] >>> N_ALPHA_SHIFT1, U1[k * N_U0 + j]);

                for (int k = 0; k < N_U0; ++k)
                    for (int j = 0; j < N_FEATURES; ++j)
                        Update(deltas0[k] >>> N_ALPHA_SHIFT0, U0[k * N_FEATURES + j]);
                        
            end // record
            $display("epoch = %0d", epoch);
        end // epoch
        
        // =====================
        // Validation after training
        // =====================

        error = 0;
        for (record = 0; record < N_V_RECORDS; ++record) begin

            // Layer 0 forward
            for (k = 0; k < N_U0; ++k) begin
                m0 = 0;
                for (j = 0; j < N_FEATURES; ++j)
                    m0 += Compute(validation_feature_at(record, j),
                                U0[k * N_FEATURES + j]);
                m0 = (m0 * N_MULT0) >>> N_BASE_SHIFT;
                models0[k] = $signed(m0[31:0]);
            end

            // Layer 1 forward
            for (k = 0; k < N_U1; ++k) begin
                m1 = 0;
                for (j = 0; j < N_U0; ++j)
                    m1 += Compute(models0[j], U1[k * N_U0 + j]);
                m1 = (m1 * N_MULT1) >>> N_BASE_SHIFT;
                models1[k] = $signed(m1[31:0]);
            end

            // Layer 2 forward
            for (k = 0; k < N_U2; ++k) begin
                m2 = 0;
                for (j = 0; j < N_U1; ++j)
                    m2 += Compute(models1[j], U2[k * N_U1 + j]);
                m2 = (m2 * N_MULT2) >>> N_BASE_SHIFT;
                models2[k] = $signed(m2[31:0]);
            end

            // Layer 3 forward
            for (k = 0; k < N_U3; ++k) begin
                m3 = 0;
                for (j = 0; j < N_U2; ++j)
                    m3 += Compute(models2[j], U3[k * N_U2 + j]);
                m3 = (m3 * N_MULT3) >>> N_BASE_SHIFT;
                models3[k] = $signed(m3[31:0]);
            end

            // Compute absolute error
            e = targets_validation[record] - models3[0];
            if (e < 0) e = -e;
            error += e;

        end // validation record
        error = error >>> 11;

        $display("Validation total error = %0d", error);       
end //initial 

    function automatic logic signed [31:0] training_feature_at(input int record, input int feature);
        training_feature_at = flat_features_training[record*N_FEATURES + feature];
    endfunction
    
    function automatic logic signed [31:0] validation_feature_at(input int record, input int feature);
        validation_feature_at = flat_features_validation[record*N_FEATURES + feature];
    endfunction

    //functions
    task automatic Initialize(
        output Function_t F,
        input  logic signed [31:0] flat_data [],  
        input  logic [31:0] xmin,
        input  logic [31:0] xmax,
        input  logic [31:0] delta_shift,
        input  int nPoints,
        input  int start
    );
        int j;
        for (j = 0; j < nPoints; j++) begin
            F.f[j] = flat_data[start + j];
        end
        F.xmin = xmin;
        F.xmax = xmax;
        F.delta_shift = delta_shift;
        F.nPoints = nPoints;
        F.index = 0;
        F.offset = 0;
    endtask
        
    function automatic logic signed [31:0] Compute(input logic signed [31:0] x_in, ref Function_t F);
        logic signed [31:0] x;
        logic signed [31:0] R;
        logic signed [31:0] diff;
        logic signed [63:0] Q;   

        begin
            x = x_in;
            if (x <= F.xmin) begin
                F.index  = 0;
                F.offset = 32'd512;
                return F.f[0];
            end
            else if (x >= F.xmax) begin
                F.index  = F.nPoints - 2;
                F.offset = (32'd1 << F.delta_shift) - 32'd512;
                return F.f[F.nPoints - 1];
            end
            else begin
                R = x - F.xmin;
                F.index  = R >>> F.delta_shift;
                F.offset = R & ((32'd1 << F.delta_shift) - 1);
                diff = F.f[F.index + 1] - F.f[F.index];
                Q    = diff * F.offset;
                Q    = Q >>> F.delta_shift;
                Q    = Q + F.f[F.index];
                return $signed(Q[31:0]);
            end
        end
    endfunction
    
    function automatic logic signed [31:0] GetDifference(ref Function_t F);
        begin
            return F.f[F.index + 1] - F.f[F.index];
        end
    endfunction
    
    task automatic Update(input logic signed [63:0] residual, ref   Function_t F);
        logic signed [63:0] prod;
        logic signed [31:0] tmp;

        begin
            prod = residual * F.offset;
            tmp  = $signed(prod >>> F.delta_shift);

            // update endpoints
            F.f[F.index + 1] = F.f[F.index + 1] + tmp;
            F.f[F.index]     = F.f[F.index]     + ($signed(residual[31:0]) - tmp);
        end
    endtask
    
    function automatic logic signed [31:0] reduce_row(input int k, input int n_cols, input logic signed [31:0] mult);
        logic signed [63:0] acc;
        acc = '0;

        for (int j = 0; j < n_cols; j++) begin
            acc += buffer[k][j];
        end

        acc = (acc * mult) >>> N_BASE_SHIFT;
        return acc[31:0];
    endfunction

endmodule
