#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <algorithm>
#include <assert.h>
#include <stdexcept>
#include <initializer_list>

class tensor
{
    friend class expression;
    friend class evaluation;

    private:
        int m_id;
        int m_dim;
        size_t m_size;
        size_t* m_shape;
        double* m_data;
        std::vector<size_t> m_locator;

    public:
        tensor(double a_value );
        tensor(int a_dim, size_t a_shape[], double a_data[]);
        ~tensor();

        int get_dim() const { return m_dim; }
        size_t get_size() const { return m_size; }
        size_t *get_shape_array() const { return m_shape; }
        double *get_data_array() const { return m_data; }

        double at(std::initializer_list<size_t> loc) const;
        void set(double value, std::initializer_list<size_t> loc);

        tensor* operator+(const tensor& a_tensor) const;
        tensor* operator-(const tensor& a_tensor) const;
        tensor* operator*(const tensor& a_tensor) const;

        tensor* copy() const;
        void print() const;

    private:
        void resetLocator();
        bool match_shape(const tensor& a_tensor) const;
        bool match_shape_multiply(const tensor& a_tensor) const;
        void checkLocation(std::initializer_list<size_t> loc) const ;
        size_t getIndex(std::initializer_list<size_t> loc) const ;
        void clear();
};
#endif