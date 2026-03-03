
            [bf16 input]
                |
         expand input row
            /           \
        [bf16]        [nvfp4]
           |            |
        gemm1         gemm1
            \          /
      [bf16 | nvfp4 (in bf16)]
                |
        activation and quant
            /           \
        [bf16]         [nvfp4]
           |            |
        gemm2         gemm2
            \          /
        (finalize fused) 
                |
             [out]




class MixPrecisionMoeFCRunner(CutlassMoeFCRunner< half, half>):
    override fusedBuildExpertMapsSortFirstToken
    override expandInputRows
    override doActivation
    

class MixPrecisionMoEFCWrapper:
    self.MixPrecisionMoeFCRunner
    self.CutlassMoeFCRunner< nvfp4, nvfp4, half, half>
    override gemm1 (takes input from self.MixPrecisionMoeFCRunner, write output to self.MixPrecisionMoeFCRunner)
    override gemm2 (takes input from self.MixPrecisionMoeFCRunner, write output to self.MixPrecisionMoeFCRunner)
