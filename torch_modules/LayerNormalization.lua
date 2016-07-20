
function nn.LayerNormalization(nOutput, bias, eps, affine)
   local eps = eps or 1e-5
   local affine = affine or true
   local bias = bias or nil 

   local input = nn.Identity()()
   local mean = nn.Mean(2)(input)
   local mean_rep = nn.Replicate(nOutput,2)(mean) 

   local input_center = nn.CSubTable()({input, mean_rep})
   local std = nn.Mean(2)(nn.Square()(input_center))
   local std_rep = nn.AddConstant(eps)(nn.Replicate(nOutput,2)(std))
   local output = nn.CDivTable()({input_center, std_rep})

   if affine == true then
      local biasTransform = nn.Add(nOutput, false)
      if bias ~=nil then
         biasTransform.bias:fill(bias)
      end
      local gainTransform = nn.CMul(nOutput)
      gainTransform.weight:fill(1.)
      output = biasTransform(gainTransform(output))
   end
   return nn.gModule({input},{output})
end



