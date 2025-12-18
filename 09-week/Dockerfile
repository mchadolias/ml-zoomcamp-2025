FROM agrigorev/model-2025-hairstyle:v1

# Install Python dependencies (keep minimal)
RUN pip install --no-cache-dir \
    onnxruntime \
    pillow \
    numpy

# Copy Lambda code into Lambda task root
COPY lambda_function.py ${LAMBDA_TASK_ROOT}/

# Lambda entrypoint
CMD ["lambda_function.lambda_handler"]
