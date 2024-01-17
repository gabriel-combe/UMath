#pragma once

#include <mkl.h>
#include <cmath>
#include <iostream>

void printMaat(float* arr) {
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			std::cout << arr[i * 3 + j] << " ";
		}
		std::cout << std::endl;
	}
}

template <typename U>
class Matrix
{
private:
	int rows, cols, lenght, ld;
	U* matrix;

	// Initialize Identity Matrix
	void Identity(const Matrix& mat) {
		for (int i = 0; i < mat.cols; i++)
			mat.matrix[i * mat.cols + i] = 1.0;
	}

	// Check if the indices of a matrix are valid
	bool checkIndex(int Rows, int Cols) {
		return (0 <= (Rows * cols + Cols) <= lenght && 0 <= Rows <= rows && 0 <= Cols <= cols);
	}

	// Check if the dimension of two Matrix match
	bool checkDimensionSum(const Matrix& mat1, const Matrix& mat2) {
		return (mat1.rows == mat2.rows && mat1.cols == mat2.cols);
	}

	// Check if the dimension of two Matrix are valid for a multiplication
	bool checkDimensionMult(const Matrix& mat1, const Matrix& mat2) {
		return (mat1.cols == mat2.rows);
	}

	// Check if the Matrix is a square Matrix
	bool checkSquareMatrix(const Matrix& mat) {
		return (mat.rows == mat.cols);
	}

public:
	// Create Matrix given the number of rows and columns
	Matrix(int Rows, int Cols) {
		rows = Rows;
		cols = Cols;
		lenght = rows * cols;
		ld = Cols;

		matrix = (U*)mkl_calloc(lenght, sizeof(U), 64);
	}

	// Create a square Matrix given the order (option to create an Identity Matrix)
	Matrix(int Size, bool identity = false) {
		rows = Size;
		cols = Size;
		lenght = rows * cols;
		ld = Size;

		matrix = (U*)mkl_calloc(lenght, sizeof(U), 64);

		if (identity)
			Identity(*this);
	}

	// Create a Matrix, with the same properties and values, given an other Matrix
	Matrix(const Matrix& mat) {
		rows = mat.rows;
		cols = mat.cols;
		lenght = rows * cols;
		ld = mat.cols;

		matrix = (U*)mkl_calloc(lenght, sizeof(U), 128);

		for (int i = 0; i < lenght; i++)
			matrix[i] = mat.matrix[i];
	}

	// Destructor Matrix
	~Matrix() {
		mkl_free(matrix);
	}
	
	// Print Matrix
	void printMat() {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				std::cout << matrix[i * cols + j] << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}

	// Get element of the matrix given a row and a column
	U GetElement(int Rows, int Cols) {
		int index = Rows * cols + Cols;
		if (checkIndex(Rows, Cols))
			return matrix[index];
		else
			return -1;
	}

	// Set element of the matrix given a row and a column
	bool SetElement(int Rows, int Cols, U value) {
		int index = Rows * cols + Cols;
		if (checkIndex(Rows, Cols)) {
			matrix[index] = value;
			return true;
		}	
		else
			return false;
	}

	// Initialize the matrix with a scalar (fill the matrix)
	void InitScalar(U value) {
		for (int i = 0; i < lenght; i++)
			matrix[i] = value;
	}

	// Initialize the matrix with an array
	void InitArray(U* array) {
		for (int i = 0; i < lenght; i++)
			matrix[i] = array[i];
	}

	// Transpose the matrix and store it inside the same matrix
	void Transpose() {
		Matrix A = *this;
		Matrix B(rows, cols);
		
		mkl_somatadd('R', 'T', 'T', cols, rows, 1, A.matrix, A.ld, 1, B.matrix, B.ld, matrix, ld);
	}

	// Inverse the matrix and store it inside the same matrix
	void Inverse() {
		if (checkSquareMatrix(*this)) {
			int n = ld;
			int* IPIV = new int[n];
			int LWORK = n * n;
			float* WORK = new float[LWORK];
			int INFO;

			sgetrf_(&n, &n, matrix, &n, IPIV, &INFO);
			sgetri_(&n, matrix, &n, IPIV, WORK, &LWORK, &INFO);

			if (INFO != 0)
				std::cout << "ERROR Matrix not inversible" << std::endl;

			delete[] IPIV;
			delete[] WORK;
		}
		std::cout << "ERROR Matrix Dimension not square" << std::endl;
	}

	Matrix PlusVectorCol(const Matrix& mat) {
		if (mat.rows == rows && mat.cols == 1) {
			Matrix Add(rows, cols);
			for (int j = 0; j < cols; j++)
				for (int i = 0; i < rows; i++)
					Add.matrix[i*cols + j] = mat.matrix[i];

			return (*this + Add);
		}
		else {
			std::cout << "ERROR input is not a vector column !" << std::endl;
			return NULL;
		}
	}

	Matrix MinusVectorCol(const Matrix& mat) {
		if (mat.rows == rows && mat.cols == 1) {
			Matrix Add(rows, cols);
			for (int j = 0; j < cols; j++)
				for (int i = 0; i < rows; i++)
					Add.matrix[i * cols + j] = mat.matrix[i];

			return (*this - Add);
		}
		else {
			std::cout << "ERROR input is not a vector column !" << std::endl;
			return NULL;
		}
	}

	Matrix PlusVectorRow(const Matrix& mat) {
		if (mat.rows == 1 && mat.cols == cols) {
			Matrix Add(rows, cols);
			for (int i = 0; i < rows; i++)
				for (int j = 0; j < cols; j++)
					Add.matrix[i * cols + j] = mat.matrix[j];

			return (*this + Add);
		}
		else {
			std::cout << "ERROR input is not a vector row !" << std::endl;
			return NULL;
		}
	}

	Matrix MinusVectorRow(const Matrix& mat) {
		if (mat.rows == 1 && mat.cols == cols) {
			Matrix Add(rows, cols);
			for (int i = 0; i < rows; i++)
				for (int j = 0; j < cols; j++)
					Add.matrix[i * cols + j] = mat.matrix[j];

			return (*this - Add);
		}
		else {
			std::cout << "ERROR input is not a vector row !" << std::endl;
			return NULL;
		}
	}

	int GetRows() {
		return rows;
	}

	int GetCols() {
		return cols;
	}

	int GetLenght() {
		return lenght;
	}

	int GetLD() {
		return ld;
	}

	// Return the transpose the matrix
	Matrix T() {
		Matrix C(cols, rows);

		mkl_somatadd('R', 'T', 'T', cols, rows, 1, matrix, ld, 0, matrix, ld, C.matrix, C.ld);

		return C;
	}

	// Return the inverse the matrix
	Matrix Inv() {
		if (checkSquareMatrix(*this)) {
			Matrix Inv = *this;
			int n = ld;
			int* IPIV = new int[n];
			int LWORK = n * n;
			float* WORK = new float[LWORK];
			int INFO;

			sgetrf_(&n, &n, Inv.matrix, &n, IPIV, &INFO);
			sgetri_(&n, Inv.matrix, &n, IPIV, WORK, &LWORK, &INFO);

			delete[] IPIV;
			delete[] WORK;

			if (INFO != 0)
				return NULL;
			else
				return Inv;
		}
		return NULL;
	}

	// Cholesky Decomposition Lower Triangular part
	Matrix CholeskyL() {
		Matrix C = *this;

		if (LAPACKE_spotrf(LAPACK_ROW_MAJOR, 'L', rows, C.matrix, ld) != 0)
			return Matrix<float>(6, true) * 0.000000001f;

		for (int i = 0; i < rows; i++)
			for (int j = 0; j < cols; j++)
				if (j > i)
					C.matrix[i * cols + j] = 0;

		return C;
	}

	// Cholesky Decomposition Upper Triangular part
	Matrix CholeskyU() {
		Matrix C = *this;

		if (LAPACKE_spotrf(LAPACK_ROW_MAJOR, 'U', rows, C.matrix, ld) != 0)
			return Matrix<float>(6,true) * 0.000000001f;

		for (int i = 0; i < rows; i++)
			for (int j = 0; j < cols; j++)
				if (j < i)
					C.matrix[i * cols + j] = 0;

		return C;
	}

	// Matrix - Matrix sum
	Matrix operator+ (const Matrix& mat) {
		if (checkDimensionSum(mat, *this)) {
			Matrix C(rows, cols);
			
			mkl_somatadd('R', 'N', 'N', rows, cols, 1, matrix, ld, 1, mat.matrix, mat.ld, C.matrix, C.ld);

			return C;
		}
	}

	// Matrix - Scalar sum
	Matrix operator+ (const U& value) {
		Matrix B(rows, cols);
		B.InitScalar(value);
		Matrix C(rows, cols);

		mkl_somatadd('R', 'N', 'N', rows, cols, 1, matrix, ld, 1, B.matrix, B.ld, C.matrix, C.ld);

		return C;
	}

	// Matrix - Matrix difference
	Matrix operator- (const Matrix& mat) {
		if (checkDimensionSum(mat, *this)) {
			Matrix C(rows, cols);

			mkl_somatadd('R', 'N', 'N', rows, cols, 1, matrix, ld, -1, mat.matrix, mat.ld, C.matrix, C.ld);

			return C;
		}
	}

	// Matrix - Scalar difference
	Matrix operator- (const U& value) {
		Matrix B(rows, cols);
		B.InitScalar(value);
		Matrix C(rows, cols);

		mkl_somatadd('R', 'N', 'N', rows, cols, 1, matrix, ld, -1, B.matrix, B.ld, C.matrix, C.ld);

		return C;
	}

	// Matrix - Matrix multiplication
	Matrix operator* (const Matrix& mat) {
		if (checkDimensionMult(*this, mat)) {
			Matrix C(rows, mat.cols);

			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, mat.cols, cols, 1, matrix, ld, mat.matrix, mat.ld, 0, C.matrix, C.ld);
		
			return C;
		}
	}

	// Matrix - Scalar multiplication
	Matrix operator* (const U& value) {
		Matrix ID(cols, true);
		Matrix C(rows, cols);

		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, cols, cols, value, matrix, ld, ID.matrix, ID.ld, 0, C.matrix, C.ld);

		return C;
	}

	// Matrix - Matrix sum
	void operator+= (const Matrix& mat) {
		if (checkDimensionSum(mat, *this)) {
			Matrix B(cols, true);

			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, cols, cols, 1, mat.matrix, mat.ld, B.matrix, B.ld, 1, matrix, ld);
		}
	}

	// Matrix - Scalar sum
	void operator+= (const U& value) {
		Matrix A(cols, true);
		Matrix B(rows, cols);
		B.InitScalar(value);
		
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, cols, cols, 1, A.matrix, A.ld, B.matrix, B.ld, 1, matrix, ld);
	}

	// Matrix - Matrix difference
	void operator-= (const Matrix& mat) {
		if (checkDimensionSum(mat, *this)) {
			Matrix B(cols, true);

			cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, cols, cols, -1, mat.matrix, mat.ld, B.matrix, B.ld, 1, matrix, ld);
		}
	}

	// Matrix - Scalar difference
	void operator-= (const U& value) {
		Matrix A(cols, true);
		Matrix B(rows, cols);
		B.InitScalar(value);

		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, cols, cols, -1, A.matrix, A.ld, B.matrix, B.ld, 1, matrix, ld);
	}

	void operator= (const Matrix& mat) {
		rows = mat.rows;
		cols = mat.cols;
		lenght = rows * cols;
		ld = mat.cols;

		for (int i = 0; i < lenght; i++)
			matrix[i] = mat.matrix[i];
	}

	bool operator== (const Matrix& mat) {
		if (rows != mat.rows || cols != mat.cols)
			return false;

		for (int i = 0; i < lenght; i++)
			if (matrix[i] != mat.matrix[i])
				return false;
		return true;
	}

	bool operator!= (const Matrix& mat) {
		return 1 - (*this == mat);
	}
};

#ifndef QUAT_h
#define QUAT_h

class Quaternion {
	friend Quaternion operator*(const Quaternion& q, const Quaternion& p) {
		return Quaternion(
			q.w * p.w - q.x * p.x - q.y * p.y - q.z * p.z,
			q.x * p.w + q.w * p.x - q.z * p.y + q.y * p.z,
			q.y * p.w + q.z * p.x + q.w * p.y - q.x * p.z,
			q.z * p.w - q.y * p.x + q.x * p.y + q.w * p.z
		);
	}

	friend Quaternion operator+(const Quaternion& q, const Quaternion& p) {
		return Quaternion(
			q.w + p.w,
			q.x + p.x,
			q.y + p.y,
			q.z + p.z
		);
	}

	friend Quaternion operator-(const Quaternion& q, const Quaternion& p) {
		return Quaternion(
			q.w - p.w,
			q.x - p.x,
			q.y - p.y,
			q.z - p.z
		);
	}

	friend Quaternion operator/(const Quaternion& q, const float& lambda) {
		return Quaternion(q.w / lambda, q.x / lambda, q.y / lambda, q.z / lambda);
	}

public:
	Quaternion(float w = 1.0f, float x = 0.0f, float y = 0.0f, float z = 0.0f) : w(w), x(x), y(y), z(z) {}

	float w;
	float x;
	float y;
	float z;

	Quaternion conj() {
		return Quaternion(this->w, -this->x, -this->y, -this->z);
	}

	Quaternion inverse() {
		float norm_sq = this->norm() * this->norm();

		return this->conj() / norm_sq;
	}

	float norm() {
		return sqrt(this->w * this->w + this->x * this->x + this->y * this->y + this->z * this->z);
	}

	bool isUnitQuat(Quaternion q) {
		if (this->norm() == 1.0f) {
			return true;
		}

		return false;
	}

	Quaternion rotQuat(Quaternion q) {
		return (q * *this) * q.inverse();
	}

	Quaternion rotQuatinv(Quaternion q) {
		return (q.inverse() * *this) * q;
	}

	void rotVect(float v[3], float result[3]) {
		float mat[3][3] = {
			{this->w * this->w + this->x * this->x - this->y * this->y - this->z * this->z, 2 * (this->x * this->y - this->w * this->z), 2 * (this->x * this->z + this->w * this->y)},
			{2 * (this->x * this->y + this->w * this->z), this->w * this->w + this->x * this->x - this->y * this->y - this->z * this->z, 2 * (this->y * this->z - this->w * this->x)},
			{2 * (this->x * this->z + this->w * this->y), 2 * (this->y * this->z + this->w * this->x), this->w * this->w - this->x * this->x - this->y * this->y + this->z * this->z}
		};
		/*
		std::cout << "[ [ " << mat[0][0] << "   " << mat[0][1] << "   " << mat[0][2] << " ]" << std::endl;
		std::cout << "  [ " << mat[1][0] << "   " << mat[1][1] << "   " << mat[1][2] << " ]" << std::endl;
		std::cout << "  [ " << mat[2][0] << "   " << mat[2][1] << "   " << mat[2][2] << " ] ]" << std::endl;*/

		result[0] = mat[0][0] * v[0] + mat[0][1] * v[1] + mat[0][2] * v[2];
		result[1] = mat[1][0] * v[0] + mat[1][1] * v[1] + mat[1][2] * v[2];
		result[2] = mat[2][0] * v[0] + mat[2][1] * v[1] + mat[2][2] * v[2];
	}

	Quaternion& operator+=(const Quaternion& q) {
		this->w += q.w;
		this->x += q.x;
		this->y += q.y;
		this->z += q.z;

		return *this;
	}

	Quaternion& operator-=(const Quaternion& q) {
		this->w -= q.w;
		this->x -= q.x;
		this->y -= q.y;
		this->z -= q.z;

		return *this;
	}

	Quaternion& operator*=(const Quaternion& q) {
		this->w = this->w * q.w - this->x * q.x - this->y * q.y - this->z * q.z;
		this->x = this->x * q.w + this->w * q.x - this->z * q.y + this->y * q.z;
		this->y = this->y * q.w + this->z * q.x + this->w * q.y - this->x * q.z;
		this->z = this->z * q.w - this->y * q.x + this->x * q.y + this->w * q.z;

		return *this;
	}

	Quaternion& operator/=(const float& lambda) {
		this->w = this->w / lambda;
		this->x = this->x / lambda;
		this->y = this->y / lambda;
		this->z = this->z / lambda;

		return *this;
	}

	bool operator==(Quaternion q) {
		if (this->w == q.w && this->x == q.x && this->y == q.y && this->z == q.z) {
			return true;
		}

		return false;
	}

	bool operator!=(Quaternion q) {
		if (*this == q) {
			return false;
		}

		return true;
	}

	Quaternion& operator=(const Quaternion& q) {
		if (&q != this) {
			this->w = q.w;
			this->x = q.x;
			this->y = q.y;
			this->z = q.z;
		}

		return *this;
	}

	friend std::ostream& operator<<(std::ostream& flux, const Quaternion& q) {
		flux << q.w << "   " << q.x << "   " << q.y << "   " << q.z;

		return flux;
	}
};

#endif

template <typename U>
Matrix<U> CovarianceRow(Matrix<U> X, Matrix<U> mean) {
	Matrix<U> Xm = X.MinusVectorCol(mean);
	return (Xm * Xm.T()) * (1.0f / (X.GetCols()));
}

template <typename U>
Matrix<U> CovarianceRow(Matrix<U> X, Matrix<U> meanx, Matrix<U> Y, Matrix<U> meany) {
	return (X.MinusVectorCol(meanx) * Y.MinusVectorCol(meany).T()) * (1.0f / (X.GetCols()));
}

template <typename U>
Matrix<U> CovarianceRow(Matrix<U> W) {
	return (W * W.T()) * (1.0f / W.GetCols());
}

template <typename U>
Matrix<U> CovarianceRow(Matrix<U> W, Matrix<U> Y, Matrix<U> meany) {
	return (W * Y.MinusVectorCol(meany).T()) * (1.0f / W.GetCols());
}

template <typename U>
Matrix<U> CovarianceCol(Matrix<U> X, Matrix<U> mean) {
	Matrix<U> Xm = X.MinusVectorRow(mean);
	return (Xm.T() * Xm) * (1.0f / (X.GetRows()));
}

template <typename U>
Matrix<U> CovarianceCol(Matrix<U> X, Matrix<U> meanx, Matrix<U> Y, Matrix<U> meany) {
	return (X.MinusVectorRow(meanx).T() * Y.MinusVectorRow(meany)) * (1.0f / (X.GetRows()));
}

template <typename U>
Matrix<U> CovarianceCol(Matrix<U> W) {
	return (W.T() * W) * (1.0f / (W.GetRows()));
}

template <typename U>
Matrix<U> CovarianceCol(Matrix<U> W, Matrix<U> Y, Matrix<U> meany) {
	return (W.T() * Y.MinusVectorCol(meany)) * (1.0f / (W.GetRows()));
}

template <typename U>
Matrix<U> MeanRow(Matrix<U> X) {
	Matrix<U> Mean(X.GetRows(), 1);
	U sum;

	for (int i = 0; i < X.GetRows(); i++)
	{
		sum = 0.0f;
		for (int j = 0; j < X.GetCols(); j++)
			sum += X.GetElement(i, j);

		Mean.SetElement(i, 0, sum / X.GetCols());
	}

	return Mean;
}

template <typename U>
Matrix<U> MeanCol(Matrix<U> X) {
	Matrix<U> Mean(1, X.GetCols());
	U sum;

	for (int j = 0; j < X.GetCols(); j++)
	{
		sum = 0.0f;
		for (int i = 0; i < X.GetRows(); i++)
			sum += X.GetElement(i, j);

		Mean.SetElement(0, j, sum / X.GetRows());
	}

	return Mean;
}

float norm_vect(float x, float y, float z) {
	return sqrt(x * x + y * y + z * z);
}

template <typename U>
Quaternion VectToQuat(Matrix<U> mat, int j = 0) {
	float norm = norm_vect(mat.GetElement(0, j), mat.GetElement(1, j), mat.GetElement(2, j));
	return Quaternion(cos(norm / 2), (mat.GetElement(0, j) / norm) * sin(norm / 2), (mat.GetElement(1, j) / norm) * sin(norm / 2), (mat.GetElement(2, j) / norm) * sin(norm / 2));
}

template <typename U>
Quaternion From_RV(Matrix<U> mat, int j, int istart = 0, float deltaT = 1) {
	float norm = norm_vect(mat.GetElement(istart, j), mat.GetElement(istart+1, j), mat.GetElement(istart+2, j));
	float theta = norm * deltaT;
	if(norm != 0.0f)
		return Quaternion (cos(theta / 2), (mat.GetElement(istart, j) / norm) * sin(theta / 2), (mat.GetElement(istart+1, j) / norm) * sin(theta / 2), (mat.GetElement(istart+2, j) / norm) * sin(theta / 2));
	else
		return Quaternion(cos(theta / 2), 0.0f, 0.0f, 0.0f);

}

Matrix<float> QuatToVectRow(Quaternion q) {
	Matrix<float> V(1, 3);
	float norm = q.norm();
	q /= norm;
	float theta = 2 * acos(q.w);
	V.SetElement(0, 0, q.x * theta); // / (sin(theta / 2)));
	V.SetElement(0, 1, q.y * theta); // / (sin(theta / 2)));
	V.SetElement(0, 2, q.z * theta); // / (sin(theta / 2)));


	return V;
}

Matrix<float> QuatToVectCol(Quaternion q) {
	Matrix<float> V(3, 1);
	float norm = q.norm();
	q /= norm;
	float theta = 2 * acos(q.w);
	V.SetElement(0, 0, q.x * theta); // / (sin(theta / 2)));
	V.SetElement(1, 0, q.y * theta); // / (sin(theta / 2)));
	V.SetElement(2, 0, q.z * theta); // / (sin(theta / 2)));

	return V;
}

Matrix<float> Sigma(Matrix<float> X, Matrix<float> P, Matrix<float> Q) {
	Matrix<float> Sig(P.GetRows() + 1, 2 * P.GetCols());

	Matrix<float> Wplus = (P + Q).CholeskyU() * sqrt(2.0f * P.GetCols());
	Matrix<float> Wminus = (P + Q).CholeskyU() * (-1.0f * sqrt(2.0f * P.GetCols()));

	Quaternion qk(X.GetElement(0, 0), X.GetElement(1, 0), X.GetElement(2, 0), X.GetElement(3, 0));

	float Wx = X.GetElement(4, 0);
	float Wy = X.GetElement(5, 0);
	float Wz = X.GetElement(6, 0);

	for (int j = 0; j < (2 * P.GetCols()); j++) {
		if (j < P.GetCols()) {
			Quaternion q = qk * From_RV(Wplus, j);

			Sig.SetElement(0, j, q.w);
			Sig.SetElement(1, j, q.x);
			Sig.SetElement(2, j, q.y);
			Sig.SetElement(3, j, q.z);

			Sig.SetElement(4, j, Wx + Wplus.GetElement(3, j));
			Sig.SetElement(5, j, Wy + Wplus.GetElement(4, j));
			Sig.SetElement(6, j, Wz + Wplus.GetElement(5, j));
		}
		else {
			Quaternion q = qk * From_RV(Wplus, j - P.GetCols());

			Sig.SetElement(0, j, q.w);
			Sig.SetElement(1, j, q.x);
			Sig.SetElement(2, j, q.y);
			Sig.SetElement(3, j, q.z);

			Sig.SetElement(4, j, Wx + Wminus.GetElement(3, j - P.GetCols()));
			Sig.SetElement(5, j, Wy + Wminus.GetElement(4, j - P.GetCols()));
			Sig.SetElement(6, j, Wz + Wminus.GetElement(5, j - P.GetCols()));
		}
	}

	return Sig;
}

Matrix<float> A(Matrix<float> Sig, float deltaT) {
	Matrix<float> Y(Sig.GetRows(), Sig.GetCols());

	for (int j = 0; j < Sig.GetCols(); j++) {
		Quaternion qk(Sig.GetElement(0, j), Sig.GetElement(1, j), Sig.GetElement(2, j), Sig.GetElement(3, j));
		Quaternion q = qk * From_RV(Sig, j, 4, deltaT);

		Y.SetElement(0, j, q.w);
		Y.SetElement(1, j, q.x);
		Y.SetElement(2, j, q.y);
		Y.SetElement(3, j, q.z);

		Y.SetElement(4, j, Sig.GetElement(4, j));
		Y.SetElement(5, j, Sig.GetElement(5, j));
		Y.SetElement(6, j, Sig.GetElement(6, j));
	}
	
	return Y;
}

Matrix<float> MeanQ(Matrix<float> X, Matrix<float> Y) {
	Matrix<float> e(3, 1);
	Matrix<float> Xm(Y.GetRows(), 1);

	Quaternion qtinv;
	Quaternion qt(X.GetElement(0, 0), X.GetElement(1, 0), X.GetElement(2, 0), X.GetElement(3, 0));

	float Wx = 0, Wy = 0, Wz = 0, iter = 0;
	
	do{
		iter += 1;
		qtinv = qt.conj();

		for (int j = 0; j < Y.GetCols(); j++) {
			Quaternion q(Y.GetElement(0, j), Y.GetElement(1, j), Y.GetElement(2, j), Y.GetElement(3, j));

			e += QuatToVectCol(q * qtinv);
		}

		e = e * (1.0f / Y.GetCols());
		qt = From_RV(e, 0) * qt;

	} while (norm_vect(e.GetElement(0, 0), e.GetElement(1, 0), e.GetElement(2, 0)) > 0.001f && iter < 10000 );

	for (int j = 0; j < Y.GetCols(); j++) {
		Wx += Y.GetElement(4, j);
		Wy += Y.GetElement(5, j);
		Wz += Y.GetElement(6, j);
	}

	Xm.SetElement(0, 0, qt.w);
	Xm.SetElement(1, 0, qt.x);
	Xm.SetElement(2, 0, qt.y);
	Xm.SetElement(3, 0, qt.z);

	Xm.SetElement(4, 0, Wx * (1.0f / Y.GetCols()));
	Xm.SetElement(5, 0, Wy * (1.0f / Y.GetCols()));
	Xm.SetElement(6, 0, Wz * (1.0f / Y.GetCols()));

	return Xm;
}

Matrix<float> DimReduc(Matrix<float> Y, Matrix<float> Xm) {
	Matrix<float> W(Y.GetRows() - 1, Y.GetCols());
	Quaternion qXm(Xm.GetElement(0,0), Xm.GetElement(1, 0), Xm.GetElement(2, 0), Xm.GetElement(3, 0));

	for (int j = 0; j < Y.GetCols(); j++) {
		Quaternion qYm(Y.GetElement(0, j), Y.GetElement(1, j), Y.GetElement(2, j), Y.GetElement(3, j));

		Matrix<float> ev = QuatToVectCol(qYm * qXm.conj());

		W.SetElement(0, j, ev.GetElement(0, 0));
		W.SetElement(1, j, ev.GetElement(1, 0));
		W.SetElement(2, j, ev.GetElement(2, 0));

		W.SetElement(3, j, Y.GetElement(4, j) - Xm.GetElement(4, 0));
		W.SetElement(4, j, Y.GetElement(5, j) - Xm.GetElement(5, 0));
		W.SetElement(5, j, Y.GetElement(6, j) - Xm.GetElement(6, 0));
	}

	return W;
}

Matrix<float> H(Matrix<float> Y, Matrix<float> g) {
	Matrix<float> Z(6, Y.GetCols());
	Quaternion qg(0.0f, g.GetElement(0, 0), g.GetElement(1, 0), g.GetElement(2, 0));

	for (int j = 0; j < Y.GetCols(); j++) {
		Quaternion qY(Y.GetElement(0, j), Y.GetElement(1, j), Y.GetElement(2, j), Y.GetElement(3, j));
		Quaternion gprime = (qY.conj() * qg) * qY;

		Z.SetElement(0, j, gprime.x);
		Z.SetElement(1, j, gprime.y);
		Z.SetElement(2, j, gprime.z);
		Z.SetElement(3, j, Y.GetElement(4, j));
		Z.SetElement(4, j, Y.GetElement(5, j));
		Z.SetElement(5, j, Y.GetElement(6, j));
	}

	return Z;
}

Matrix<float> KalmanGain(Matrix<float> Pxz, Matrix<float> Pvv) {
	Matrix<float> K(Pxz.GetRows(), Pvv.GetCols());
	Matrix<float> Pinv = Pvv.Inv();

	if (Pinv != NULL)
		K = Pxz * Pinv;

	return K;
}

Matrix<float> StateUpdate(Matrix<float> Xm, Matrix<float> KV) {
	Matrix<float> X(Xm.GetRows(), 1);

	Quaternion qx(Xm.GetElement(0, 0), Xm.GetElement(1, 0), Xm.GetElement(2, 0), Xm.GetElement(3, 0));

	Quaternion q = qx * From_RV(KV, 0);

	X.SetElement(0, 0, q.w);
	X.SetElement(1, 0, q.x);
	X.SetElement(2, 0, q.y);
	X.SetElement(3, 0, q.z);

	X.SetElement(4, 0, Xm.GetElement(4, 0) + KV.GetElement(3, 0));
	X.SetElement(5, 0, Xm.GetElement(5, 0) + KV.GetElement(4, 0));
	X.SetElement(6, 0, Xm.GetElement(6, 0) + KV.GetElement(5, 0));

	return X;
}

void UKF(Matrix<float>& X, Matrix<float>& P, Matrix<float>& Zk, Matrix<float>& g, float deltaT, Matrix<float>& Q, Matrix<float>& R) {

	Matrix<float> Sig = Sigma(X, P, Q);

	Matrix<float> Y = A(Sig, deltaT);

	Matrix<float> Ym = MeanQ(X, Y);

	Matrix<float> W = DimReduc(Y, Ym);

	Matrix<float> Pm = CovarianceRow(W);

	Matrix<float> Z = H(Sig, g);

	Matrix<float> Zm = MeanRow(Z);

	Matrix<float> Pvv = CovarianceRow(Z, Zm) + R;

	Matrix<float> Pxz = CovarianceRow(W, Z, Zm);

	Matrix<float> K = KalmanGain(Pxz, Pvv);

	X = StateUpdate(Ym, (K * (Zk - Zm)));

	P = Pm - ((K * Pvv) * K.T());


}

Matrix<float> QTR(Matrix<float> State) {
	Matrix<float> rot(1, 3);
	Quaternion q(State.GetElement(0, 0), State.GetElement(1, 0), State.GetElement(2, 0), State.GetElement(3, 0));

	rot.SetElement(0, 0, atan2(2.0f * (q.y * q.z + q.w * q.x), q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z));
	rot.SetElement(0, 1, asin(-2.0f * (q.x * q.z - q.w * q.y)));
	rot.SetElement(0, 2, atan2(2.0f * (q.x * q.y + q.w * q.z), q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z));

	return rot;
}