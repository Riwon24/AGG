"""env_smoke_test: exp/.venv 에 설치한 패키지가 이 PC(Windows 애플리케이션 제어 정책 포함)에서 실제로 로드/실행되는지 확인한다.

실패(ImportError, DLL load failed 등)하면 트레이스백을 출력하고 종료 코드 3 으로 끝난다. 우회하지 않는다.
"""
import sys
import traceback


def main():
    try:
        import numpy as np
        import scipy
        import PIL
        import yaml
        import matplotlib
        import sklearn
        import torch
        import torchvision
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.preprocessing import StandardScaler
        print("python", sys.version.split()[0])
        for m in (np, scipy, PIL, yaml, matplotlib, sklearn, torch, torchvision):
            print(f"{m.__name__:<12} {getattr(m, '__version__', '?')}")
        print("torch.version.cuda:", torch.version.cuda)
        print("cuda 사용 가능:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
            a = torch.randn(256, 256, device="cuda"); b = a @ a
            torch.cuda.synchronize()
            print("cuda matmul OK, 합:", float(b.sum().cpu()) == float(b.sum().cpu()))
        rng = np.random.RandomState(0)
        X = rng.randn(60, 16); y = np.repeat([0, 1, 2], 20)
        Z = StandardScaler().fit_transform(X)
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(Z, y)
        KNeighborsClassifier(n_neighbors=5, metric="cosine", weights="distance").fit(X, y)
        print("sklearn logreg/knn fit OK")
        print("SMOKE_TEST_OK")
    except BaseException:
        traceback.print_exc()
        print("SMOKE_TEST_FAILED")
        sys.exit(3)


if __name__ == "__main__":
    main()
