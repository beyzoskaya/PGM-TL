from flip_hf import Thermostability, SecondaryStructure, CloningCLF

if __name__ == "__main__":
    print("Testing Thermostability dataset...")
    ts = Thermostability(verbose=1)
    print(f"Number of samples: {len(ts)}")
    sample = ts[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Sequence length: {len(sample['sequence'])}")
    print(f"Target example: {sample['targets']}")

    print("\nTesting SecondaryStructure dataset...")
    ss = SecondaryStructure(verbose=1)
    print(f"Number of samples: {len(ss)}")
    sample = ss[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Sequence length: {len(sample['sequence'])}")
    print(f"Target example (first 30 residues): {sample['targets']['target'][:30]}")

    print("\nTesting CloningCLF dataset...")
    clf = CloningCLF(verbose=1)
    print(f"Number of samples: {len(clf)}")
    sample = clf[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Sequence length: {len(sample['sequence'])}")
    print(f"Target example: {sample['targets']}")
