#include "tensor.h"
#include <stdio.h>

static int ten_ids;
tensor::tensor(double a_value):tensor(0,nullptr, new double[1]{a_value}){};
tensor::tensor(
    int a_dim, 
    size_t a_shape[], 
    double a_data[] )
:m_id(ten_ids++),m_dim(a_dim), m_shape(nullptr), m_data(nullptr), m_size(1)
{
    m_locator.clear();
    m_locator.push_back(m_size);
    if (m_dim > 0 && a_shape != nullptr)
    {
        m_shape=new size_t[m_dim];
        for (size_t i = 0; i < m_dim; i++)
        {
            m_shape[m_dim-1-i]=a_shape[m_dim-1-i];
            m_size*=m_shape[m_dim-1-i];
            m_locator.push_back(m_size);
        }
    }
    m_locator.pop_back();
    std::reverse(m_locator.begin(),m_locator.end());

    if (m_size > 0 && a_data != nullptr)
    {
        m_data = new double[m_size];
        for (size_t i = 0; i < m_size; i++)
        {
            m_data[i] = a_data[i];
        }
    }
    printf("[Tensor/%d]: Created\n",this->m_id);
    print();
    printf("[Tensor/%d]: Locator: [",this->m_id);
    for (size_t i:m_locator)
    {
        printf("%zu, ",i);
    }
    printf("]\n",this->m_id);
}
tensor::~tensor(){
    if (m_shape != nullptr)
    {
        delete(m_shape);
        m_shape = nullptr;
    }
    if (m_data != nullptr)
    {
        delete(m_data);
        m_data = nullptr;
    }
    printf("[Tensor/%d]: Removed\n",this->m_id);
}

double tensor::at(std::initializer_list<size_t> loc) const {
    size_t index=getIndex(loc);
    return m_data[index];
}
void tensor::set(double value, std::initializer_list<size_t> loc){
    size_t index=getIndex(loc);
    m_data[index]=value;
}

tensor* tensor::operator+(const tensor& a_tensor) const {
    if (!this->match_shape(a_tensor))
    {
        printf("[Tensor/%d]: Shape mismatch\n", m_id);
        throw std::out_of_range("Tensor shape mismatch.");
    }
    tensor* result=this->copy();
    for (size_t i = 0; i < result->m_size; i++)
    {
        result->m_data[i]+=a_tensor.m_data[i];
    }
    return result;
}
tensor* tensor::operator-(const tensor& a_tensor) const {
    if (!this->match_shape(a_tensor))
    {
        printf("[Tensor/%d]: Shape mismatch\n", m_id);
        throw std::out_of_range("Tensor shape mismatch.");
    }
    tensor* result=this->copy();
    for (size_t i = 0; i < result->m_size; i++)
    {
        result->m_data[i]-=a_tensor.m_data[i];
    }
    return result;
}
tensor* tensor::operator*(const tensor& a_tensor) const {
    int dim;
    size_t size;
    size_t* shape;
    double* data;
    tensor* result;
    if (this->m_dim<=0)
    {
        result=a_tensor.copy();
        for (size_t i = 0; i < result->m_size; i++)
        {
            result->m_data[i]*=this->m_data[0];
        }
    }
    else if (a_tensor.get_dim()<=0)
    {
        result=this->copy();
        for (size_t i = 0; i < result->m_size; i++)
        {
            result->m_data[i]*=a_tensor.m_data[0];
        }
    }
    else if(this->m_dim==2&&
            a_tensor.m_dim==2&&
            this->m_shape[1]==a_tensor.m_shape[0])
    {
        printf("[Tensor/%d]: Matrix multiply\n", m_id);
        dim=2;
        size = this->m_shape[0]*a_tensor.m_shape[1];
        shape = new size_t[2]{this->m_shape[0], a_tensor.m_shape[1]};
        data = new double[size];

        result = new tensor(0);
        result->clear();
        result->m_dim=dim;
        result->m_size=size;
        delete(result->m_shape);
        result->m_shape = shape;
        delete(result->m_data);
        result->m_data = data;
        result->resetLocator();

        printf("[Tensor/%d]: Correcting data\n", m_id);
        double value;
        for (size_t x = 0; x < this->m_shape[0]; x++)
        {
            for (size_t y = 0; y < a_tensor.m_shape[1]; y++)
            {
                value=0;
                for (size_t z = 0; z < this->m_shape[1]; z++)
                {
                    value+=this->at({x, z})*a_tensor.at({z, y});
                }
                printf("[Tensor/%d]: Setting data to {%d,%d}:%d\n", m_id,x,y,result->getIndex({x,y}));
                result->set(value, { x, y });
            }
        }
    }
    else
    {
        throw std::out_of_range("Tensor shape mismatch.");
    }
    return result;
}


tensor* tensor::copy() const{
    tensor* t=new tensor(this->m_dim, this->m_shape, this->m_data);
    return t;
}

void tensor::print() const{
    printf("[Tensor/%d]: Dim:%d",this->m_id, this->m_dim);
    if (m_dim > 0 && m_shape != nullptr)
    {
        printf(", Shape:[ ");
        for (size_t i = 0; i < m_dim; i++)
        {
            printf("%d, ",m_shape[i]);
        }
        printf("]");
    }
    printf("\n");
    if (m_size < 10)
    {
        printf("[Tensor/%d]: [ ",this->m_id);
        for (size_t i = 0; i < m_size; i++)
        {
            printf("%f, ", m_data[i]);
        }
        printf("]\n");
    }
}

void tensor::resetLocator(){
    printf("[Tensor/%d]: Reset locator\n",this->m_id);
    m_locator.clear();
    m_size=1;
    m_locator.push_back(m_size);
    if (m_dim > 0 && m_shape != nullptr)
    {
        for (size_t i = 0; i < m_dim; i++)
        {
            m_size*=m_shape[m_dim-1-i];
            m_locator.push_back(m_size);
        }
    }
    m_locator.pop_back();
    std::reverse(m_locator.begin(),m_locator.end());
    printf("[Tensor/%d]: Locator: [",this->m_id);
    for (size_t i:m_locator)
    {
        printf("%zu, ",i);
    }
    printf("]\n",this->m_id);
}

bool tensor::match_shape(const tensor& a_tensor) const {
    bool match=false;
    if (this->m_dim==a_tensor.m_dim)
    {
        if (m_dim==0)
        {
            match = true;
        }
        else
        {
            if (m_shape == nullptr || a_tensor.m_shape == nullptr)
            {
                
            }
            for (int i = 0; i < this->m_dim; i++)
            {
                if (m_shape[i]!=a_tensor.m_shape[i])
                {
                    match=true;
                    break;
                }
            }
            match=!match;
        }
    }
    return match;
}

bool tensor::match_shape_multiply(const tensor& a_tensor) const {
    bool match=false;
    if (this->m_dim==a_tensor.m_dim)
    {
        for (int i = 0; i < this->m_dim; i++)
        {
            if (m_shape[i]!=a_tensor.m_shape[i])
            {
                match=true;
                break;
            }
        }
        match=!match;
    }
    return match;
}

void tensor::checkLocation(std::initializer_list<size_t> loc) const{
    if (m_dim != loc.size())
    {
        printf("[Tensor/%d]: Location out of range. Dim: %d. Loc: ", this->m_id, m_dim);
        for (auto l=loc.begin(); l != loc.end(); l++)
        {
            printf("%d, ",l);
        }
        printf("\n");
        throw std::out_of_range("Location out of tensor range");
    }
    auto iter = loc.begin();
    for (size_t i = 0; i < m_dim; i++)
    {
        if (m_shape[i]<(*iter))
        {
            printf("[Tensor/%d]: Location out of range. Dim: %d. Loc: ", this->m_id, m_dim);
            for (auto l=loc.begin(); l != loc.end(); l++)
            {
                printf("%d, ",l);
            }
            printf("\n");
            throw std::out_of_range("Location out of tensor range");
        }
        iter++;
    }
}
size_t tensor::getIndex(std::initializer_list<size_t> loc) const {
    checkLocation(loc);
    size_t index=0;
    auto iter = loc.begin();
    for (size_t i = 0; i < m_dim; i++)
    {
        index+=(*iter)*m_locator[i];
        iter++;
    }
    return index;
}

void tensor::clear(){
    printf("[Tensor/%d]: Clearing\n", this->m_id);
    if (m_shape != nullptr)
    {
        delete(m_shape);
        m_shape = nullptr;
    }
    if (m_data != nullptr)
    {
        delete(m_data);
        m_data = nullptr;
    }
}
