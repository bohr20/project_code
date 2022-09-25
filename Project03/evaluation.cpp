#include <assert.h>
#include "evaluation.h"
#include <stdio.h>

static int eval_ids;

evaluation::evaluation(
    expression* output,
    std::vector<expression*> inputs)
    :m_id(eval_ids++),m_output(output),m_inputs(inputs),m_result(nullptr)
{
    clear_runtime();
    printf("[Evaluation/%d]: Created\n",m_id);
}

evaluation::~evaluation(){
    if (m_result != nullptr)
    {
        delete(m_result);
        m_result=nullptr;
    }
    clear_runtime();
    m_output->excute();
    delete(m_output);
    printf("[Evaluation/%d]: Cleared\n",m_id);
}

void evaluation::clear_runtime(){
    printf("[Evaluation/%d]: Clearing runtime\n",m_id);
    for (auto pair : m_params)
    {
        delete(pair.second);
    }
    m_params.clear();
    m_output->clear_all();
}

int evaluation::add_kwargs_double(
    const char *key,
    double value)
{
    printf("[Evaluation/%d]: Adding param double: %s\n",m_id, key);
    if (this->m_params.count(key)>0)
    {
        this->m_params[key]->clear();
        delete(this->m_params[key]);
    }
    this->m_params[key]=new tensor(value);  
    return 0;
}

int evaluation::add_kwargs_ndarray(
    const char *key,
    int dim,
    size_t shape[],
    double data[])
{
    printf("[Evaluation/%d]: Adding param ndarray: %s\n",m_id, key);
    if (this->m_params.count(key)>0)
    {
        this->m_params[key]->clear();
        delete(this->m_params[key]);
    }
    this->m_params[key]=new tensor(dim, shape, data);  
    return 0;
}

int evaluation::execute()
{
    if (m_result != nullptr)
    {
        m_result->clear();
        delete(m_result);
        m_result=nullptr;
    }
    printf("[Evaluation/%d]: Executing\n",m_id);
    for (auto exp : m_inputs)
    {
        if (exp->getType().compare("Input")>=0 || exp->getName().compare("Input2d")>=0)
        {
            exp->set_value(m_params[exp->getName()]->copy());
        }
        exp->post();
    }
    if (m_output->m_result==nullptr)
    {
        printf("[Evaluation/%d]: Unexpected exception, no result received\n",m_id);
    }
    else
    {
        this->m_result=m_output->m_result->copy();
        this->m_result->print();
        clear_runtime();
    }
    
    return 0;
}

tensor* evaluation::get_result()
{
    return this->m_result;
}
