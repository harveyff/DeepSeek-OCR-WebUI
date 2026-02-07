"""CUDA Backend for NVIDIA GPUs"""
import os
import warnings
from transformers import AutoProcessor, AutoModel
import torch

class CUDABackend:
    def __init__(self, model_path: str = "deepseek-ai/DeepSeek-OCR"):
        self.model_path = model_path
        self.revision = "1e3401a3d4603e9e71ea0ec850bfead602191ec4"  # MPS support commit
        self.model = None
        self.processor = None
        
    @staticmethod
    def get_optimal_dtype():
        """Get optimal dtype based on GPU capability"""
        if not torch.cuda.is_available():
            return torch.float32
        
        # Check if GPU supports bfloat16 (compute capability >= 8.0)
        capability = torch.cuda.get_device_capability()
        if capability[0] >= 8:
            # Ampere and newer (RTX 30xx, A100, etc.)
            return torch.bfloat16
        else:
            # Older GPUs (RTX 20xx, GTX 10xx, etc.) - use float16
            print(f"⚠️ GPU compute capability {capability[0]}.{capability[1]} < 8.0, using float16 instead of bfloat16")
            return torch.float16
        
    def load_model(self, source: str = "huggingface", timeout: int = 300):
        """Load CUDA model"""
        try:
            # Verify CUDA is available before proceeding
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available in PyTorch. Please ensure PyTorch is compiled with CUDA support.")
            
            print(f"📦 Loading DeepSeek-OCR on CUDA")
            print(f"🔍 CUDA Device: {torch.cuda.get_device_name(0)}")
            print(f"🔍 CUDA Version: {torch.version.cuda}")
            
            if source == "modelscope":
                # ModelScope fallback for China
                from modelscope import snapshot_download
                cache_dir = os.environ.get('MODELSCOPE_CACHE', '/app/models/modelscope')
                print(f"📥 Downloading from ModelScope to {cache_dir}...")
                try:
                    local_path = snapshot_download(
                        model_id=self.model_path,
                        cache_dir=cache_dir,
                        revision='master',
                        local_files_only=False  # Allow download if not cached
                    )
                    model_path = local_path
                    revision = None
                    print(f"✅ ModelScope download completed: {local_path}")
                except Exception as e:
                    print(f"❌ ModelScope download failed: {e}")
                    raise
            else:
                os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = str(timeout)
                model_path = self.model_path
                revision = self.revision
                print(f"📥 Loading from HuggingFace: {model_path}")
                if revision:
                    print(f"   Revision: {revision}")
                
                # Check cache directory
                cache_dir = os.environ.get('HF_HOME', os.environ.get('TRANSFORMERS_CACHE', '/app/models'))
                print(f"   Cache directory: {cache_dir}")
                
                # Check if model might be cached
                from pathlib import Path
                cache_path = Path(cache_dir) / "hub" / f"models--{model_path.replace('/', '--')}"
                if cache_path.exists():
                    print(f"   ✅ Found cached model at: {cache_path}")
                    print(f"   ⏳ Loading from cache (should be faster)...")
                else:
                    print(f"   ⚠️  Model not found in cache")
                    print(f"   ⏳ Downloading model for the first time (~8GB, may take 5-15 minutes)...")
                    print(f"   💡 Tip: Model will be cached for future use")
            
            print("📦 Step 1/2: Loading processor...")
            import time
            start_time = time.time()
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                revision=revision,
                trust_remote_code=True
            )
            elapsed = time.time() - start_time
            print(f"✅ Processor loaded successfully (took {elapsed:.1f}s)")
            
            # Use optimal dtype based on GPU capability
            optimal_dtype = self.get_optimal_dtype()
            print(f"📊 Using dtype: {optimal_dtype}")
            
            print("📦 Step 2/2: Loading model (this may take 2-10 minutes)...")
            print("   ⏳ Downloading/loading model files...")
            start_time = time.time()
            # Suppress model type mismatch warning (deepseek_vl_v2 vs DeepseekOCR)
            # This is a known compatibility warning that doesn't affect functionality
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*You are using a model of type.*")
                self.model = AutoModel.from_pretrained(
                    model_path,
                    revision=revision,
                    trust_remote_code=True,
                    torch_dtype=optimal_dtype,
                    low_cpu_mem_usage=True
                )
            elapsed = time.time() - start_time
            print(f"   ✅ Model files loaded (took {elapsed:.1f}s)")
            print("   ⏳ Moving model to CUDA device...")
            start_time = time.time()
            self.model = self.model.to("cuda")
            elapsed = time.time() - start_time
            print(f"   ✅ Model moved to CUDA (took {elapsed:.1f}s)")
            
            print("   ⏳ Converting model dtype...")
            # Ensure all parameters use consistent dtype to avoid mismatch errors
            # Convert model to the optimal dtype if needed
            if optimal_dtype == torch.bfloat16:
                # For bfloat16, ensure all layers are properly converted
                self.model = self.model.to(dtype=optimal_dtype)
            elif optimal_dtype == torch.float16:
                self.model = self.model.half()
            # float32 doesn't need conversion
            
            print("   ⏳ Setting model to evaluation mode...")
            self.model.eval()
            print(f"✅ Model loaded successfully on CUDA from {source} (dtype: {optimal_dtype})")
            return True
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise
    
    def infer(self, prompt: str, image_path: str, **kwargs) -> str:
        """Run inference on CUDA"""
        try:
            result = self.model.infer(
                tokenizer=self.processor,
                prompt=prompt,
                image_file=image_path,
                output_path='./output',
                base_size=1024,
                image_size=640,
                crop_mode=True,
                test_compress=False,
                save_results=False,
                eval_mode=True
            )
            return result if result else ""
        except Exception as e:
            print(f"❌ Inference failed: {e}")
            raise
    
    @staticmethod
    def is_available() -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
