#!/usr/bin/env python3
"""
Pre-download all required models for the document retrieval pipeline.
Run this script once during setup to avoid download delays during runtime.

Usage:
    python scripts/download_models.py [OPTIONS]

Options:
    --all              Download all models (default)
    --embeddings       Download embedding models only
    --reranker         Download reranking models only
    --spacy            Download Spacy models only
    --upgrade          Download upgraded embedding models (BAAI/bge-large)
    --verify           Verify downloads without downloading
    --help             Show this help message

Examples:
    python scripts/download_models.py
    python scripts/download_models.py --embeddings --reranker
    python scripts/download_models.py --verify
"""
import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "python_services"))

def download_huggingface_models(include_upgrade=False):
    """Download Hugging Face models"""
    print("\n" + "=" * 60)
    print("Downloading Hugging Face Models")
    print("=" * 60)

    try:
        from sentence_transformers import SentenceTransformer, CrossEncoder
    except ImportError:
        print("❌ Error: sentence-transformers not installed")
        print("   Run: pip install sentence-transformers")
        return False

    models_to_download = []

    # Current embedding model (required)
    models_to_download.append({
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "type": "embedding",
        "size": "80 MB",
        "required": True
    })

    # Cross-encoder for reranking (required for Phase 2)
    models_to_download.append({
        "name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "type": "cross-encoder",
        "size": "90 MB",
        "required": True
    })

    # Upgraded embedding model (optional)
    if include_upgrade:
        models_to_download.append({
            "name": "BAAI/bge-large-en-v1.5",
            "type": "embedding",
            "size": "1.3 GB",
            "required": False
        })

    for model_info in models_to_download:
        try:
            print(f"\n📥 Downloading: {model_info['name']}")
            print(f"   Type: {model_info['type']}")
            print(f"   Size: ~{model_info['size']}")

            if model_info['type'] == 'cross-encoder':
                model = CrossEncoder(model_info['name'])
            else:
                model = SentenceTransformer(model_info['name'])

            print(f"   ✓ Successfully downloaded and cached")

        except Exception as e:
            print(f"   ❌ Error downloading {model_info['name']}: {e}")
            if model_info['required']:
                return False

    print("\n✓ Hugging Face models downloaded successfully!")
    return True

def download_spacy_models():
    """Download Spacy models"""
    print("\n" + "=" * 60)
    print("Downloading Spacy Models")
    print("=" * 60)

    try:
        import spacy
        import spacy.cli
    except ImportError:
        print("❌ Error: spacy not installed")
        print("   Run: pip install spacy")
        return False

    models_to_download = [
        {
            "name": "en_core_web_sm",
            "size": "12 MB",
            "description": "English language model (small, fast)"
        }
    ]

    for model_info in models_to_download:
        try:
            print(f"\n📥 Downloading: {model_info['name']}")
            print(f"   Size: ~{model_info['size']}")
            print(f"   Description: {model_info['description']}")

            # Check if already installed
            try:
                spacy.load(model_info['name'])
                print(f"   ✓ Already installed, skipping")
                continue
            except OSError:
                pass

            # Download model
            spacy.cli.download(model_info['name'])
            print(f"   ✓ Successfully downloaded")

        except Exception as e:
            print(f"   ❌ Error downloading {model_info['name']}: {e}")
            return False

    print("\n✓ Spacy models downloaded successfully!")
    return True

def verify_downloads(check_upgrade=False):
    """Verify all models are accessible"""
    print("\n" + "=" * 60)
    print("Verifying Model Downloads")
    print("=" * 60)

    all_ok = True

    # Check sentence-transformers
    try:
        print("\n🔍 Checking: Embedding model (all-MiniLM-L6-v2)")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        test_embedding = model.encode("test query")
        print(f"   ✓ Working (dimension: {len(test_embedding)})")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        all_ok = False

    # Check cross-encoder
    try:
        print("\n🔍 Checking: Cross-encoder reranker (ms-marco-MiniLM-L-6-v2)")
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        test_score = reranker.predict([("query", "document")])
        print(f"   ✓ Working (test score: {test_score[0]:.4f})")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        all_ok = False

    # Check upgraded embedding model if requested
    if check_upgrade:
        try:
            print("\n🔍 Checking: Upgraded embedding model (BAAI/bge-large-en-v1.5)")
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('BAAI/bge-large-en-v1.5')
            test_embedding = model.encode("test query")
            print(f"   ✓ Working (dimension: {len(test_embedding)})")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            print(f"   ℹ️  This is optional - you can skip if not using")

    # Check spacy
    try:
        print("\n🔍 Checking: Spacy model (en_core_web_sm)")
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp("test text")
        print(f"   ✓ Working ({len(nlp.pipe_names)} pipeline components)")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        all_ok = False

    # Check torch (required for models)
    try:
        print("\n🔍 Checking: PyTorch installation")
        import torch
        print(f"   ✓ PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"   ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print(f"   ℹ️  CUDA not available (CPU mode only)")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        all_ok = False

    # Summary
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ All models verified successfully!")
        print("\nYour system is ready for the document retrieval pipeline.")
    else:
        print("❌ Some models failed verification.")
        print("\nPlease fix the errors above and run again.")

    print("=" * 60)

    return all_ok

def show_disk_usage():
    """Show estimated disk usage for models"""
    print("\n" + "=" * 60)
    print("Estimated Disk Usage")
    print("=" * 60)

    usage = [
        ("Embedding model (all-MiniLM-L6-v2)", "80 MB", True),
        ("Cross-encoder (ms-marco-MiniLM-L-6-v2)", "90 MB", True),
        ("Spacy model (en_core_web_sm)", "12 MB", True),
        ("PyTorch + dependencies", "~2 GB", True),
        ("", "", False),
        ("TOTAL (Required)", "~2.2 GB", True),
        ("", "", False),
        ("Optional: Upgraded embedding (BAAI/bge-large)", "1.3 GB", False),
        ("Optional: Larger Spacy model (en_core_web_lg)", "560 MB", False),
    ]

    print("\n{:<50} {:>15} {:>10}".format("Component", "Size", "Required"))
    print("-" * 77)
    for name, size, required in usage:
        if not name:
            print()
            continue
        req_mark = "✓" if required else " "
        print(f"{name:<50} {size:>15} {req_mark:>10}")

    print("\n" + "=" * 60)

def main():
    """Main download process"""
    parser = argparse.ArgumentParser(
        description="Download models for document retrieval pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--all', action='store_true', default=True,
                       help='Download all required models (default)')
    parser.add_argument('--embeddings', action='store_true',
                       help='Download embedding models only')
    parser.add_argument('--reranker', action='store_true',
                       help='Download reranking models only')
    parser.add_argument('--spacy', action='store_true',
                       help='Download Spacy models only')
    parser.add_argument('--upgrade', action='store_true',
                       help='Download upgraded embedding models (BAAI/bge-large)')
    parser.add_argument('--verify', action='store_true',
                       help='Verify downloads without downloading')
    parser.add_argument('--disk-usage', action='store_true',
                       help='Show estimated disk usage')

    args = parser.parse_args()

    # Show header
    print("=" * 60)
    print("Document Retrieval Pipeline - Model Download")
    print("=" * 60)
    print("\nThis script will download the following models:")
    print("  • Embedding model (sentence-transformers)")
    print("  • Cross-encoder reranker")
    print("  • Spacy language model")
    if args.upgrade:
        print("  • Upgraded embedding model (BAAI/bge-large)")
    print("\nEstimated disk space: ~2.2 GB")
    if args.upgrade:
        print("  (+ 1.3 GB for upgraded model)")

    # Show disk usage if requested
    if args.disk_usage:
        show_disk_usage()
        return 0

    # Verify only mode
    if args.verify:
        success = verify_downloads(check_upgrade=args.upgrade)
        return 0 if success else 1

    # Determine what to download
    download_hf = args.all or args.embeddings or args.reranker
    download_sp = args.all or args.spacy

    # If specific options are selected, override --all
    if args.embeddings or args.reranker or args.spacy:
        download_hf = args.embeddings or args.reranker
        download_sp = args.spacy

    try:
        # Download models
        success = True

        if download_hf:
            if not download_huggingface_models(include_upgrade=args.upgrade):
                success = False

        if download_sp:
            if not download_spacy_models():
                success = False

        # Verify downloads
        if success:
            print("\n")
            if verify_downloads(check_upgrade=args.upgrade):
                print("\n🎉 Setup complete! All models are ready.")
                print("\nNext steps:")
                print("  1. Configure environment variables (see docs/operations/tooling.md)")
                print("  2. Install and start Redis server")
                print("  3. Create MongoDB indexes")
                print("  4. Run the pipeline tests")
                return 0
            else:
                print("\n⚠️  Downloads completed but verification failed.")
                print("   Please check the errors above.")
                return 1
        else:
            print("\n❌ Download failed. Please check errors above.")
            return 1

    except KeyboardInterrupt:
        print("\n\n⚠️  Download cancelled by user.")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
