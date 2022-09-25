#include "program.h"
#include "evaluation.h"
#include <stdio.h>

#include <vector>

static int prog_ids;

program::program():m_id(prog_ids++)
{
    this->m_id_exp_map.clear();
    this->m_inputs.clear();
    printf("[Program:%d]: Created.\n",this->m_id);
}
program::~program(){
    this->m_id_exp_map.clear();
    this->m_inputs.clear();
    printf("[Program:%d]: Cleared.\n",this->m_id);
}

void program::append_expression(
    int expr_id,
    const char *op_name,
    const char *op_type,
    int inputs[],
    int num_inputs)
{
    printf("[Program:%d]: Appending expression:%d\n", this->m_id, expr_id);
    if (this->m_id_exp_map.count(expr_id)<=0)
    {
        std::vector<expression*> exp_inputs;
        for (size_t i = 0; i < num_inputs; i++)
        {
            printf("[Program:%d]: Sub expression:%d\n", this->m_id, inputs[i]);
            exp_inputs.push_back(this->m_id_exp_map[inputs[i]]);
        }
        exp_inputs.resize(num_inputs);
        this->m_current_expression=new expression(expr_id, op_name, op_type, exp_inputs);
        printf("[Program:%d]: Expression %d created\n", this->m_id, expr_id);
        this->m_id_exp_map.insert(std::pair<int, expression*>(expr_id, this->m_current_expression));
        printf("[Program:%d]: Expression %d registered\n", this->m_id, expr_id);
        if (num_inputs==0)
        {
            printf("[Program:%d]: Registering input Expression %d\n", this->m_id, expr_id);
            bool add=true;
            for (auto iter=m_inputs.begin();iter !=m_inputs.end();iter++)
            {
                if ((*iter)->getID()==this->m_current_expression->getID())
                {
                    printf("[Program:%d]: Expression %d already exist in inputs\n", this->m_id, expr_id);
                    add=false;
                    break;
                }
            }
            if (add)
            {
                printf("[Program:%d]: Adding input Expression %d to vector\n", this->m_id, expr_id);
                this->m_inputs.push_back(this->m_current_expression);
                printf("[Program:%d]: Registered input Expression %d\n", this->m_id, expr_id);
            }
        }
    }
    else
    {
        this->m_current_expression=this->m_id_exp_map[expr_id];
    }
}

int program::add_op_param_double(
    const char *key,
    double value)
{
    printf("[Program/%d]: Appending param\n", this->m_id);
    if (this->m_current_expression==nullptr)
    {
        printf("[Program/%d]: Null current when add_op_param_double\n", this->m_id);
        return 1;
    }
    return this->m_current_expression->add_op_param_double(key, value);
}

int program::add_op_param_ndarray(
    const char *key,
    int dim,
    size_t shape[],
    double data[])
{
    printf("[Program/%d]: Appending param\n", this->m_id);
    if (this->m_current_expression==nullptr)
    {
        printf("[Program/%d]: Null current when add_op_param_ndarray\n", this->m_id);
        return 2;
    }
    return this->m_current_expression->add_op_param_ndarray(key, dim, shape, data);
}

evaluation *program::build()
{
    printf("[Program/%d]: Building\n", this->m_id);
    return new evaluation(this->m_current_expression, this->m_inputs);
}
