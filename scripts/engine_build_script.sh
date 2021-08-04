# Static shape
./bin/trtexec --explicitBatch \
          --onnx=./bin/glare_bestmodel_35_11_train_37_28.onnx \
          --verbose=true \
          --workspace=4096 \
          --fp16 \
          --saveEngine=./bin/glare_bestmodel_35_11_train_37_28.engine