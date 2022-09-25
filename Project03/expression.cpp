#include "expression.h"
#include <stdio.h>


expression::expression(){};
expression::expression(
    int id,
    const char *op_name,
    const char *op_type,
    std::vector<expression*> inputs)
    :m_id(id),m_name(op_name),m_type(op_type),m_inputs(inputs),m_result(nullptr)
{
    printf("[Expression/%d/%s]: Creating\n",m_id,m_name.length()==0?m_type.data():(m_type+"/"+m_name).data());
    bool add;
    for (auto iter = m_inputs.begin(); iter != m_inputs.end(); iter++)
    {
        add=true;
        for (auto iter_ins = (*iter)->m_outputs.begin(); iter_ins != (*iter)->m_outputs.end(); iter_ins++)
        {
            if ((*iter_ins)->m_id == this->m_id)
            {
                add=false;
                break;
            }
        }
        if (add)
        {
            (*iter)->m_outputs.push_back(this);
        }
    }
    for (auto pair : m_params)
    {
        pair.second->clear();
        delete(pair.second);
    }
    m_params.clear();
    if (this->m_result != nullptr)
    {
        this->m_result->clear();
        delete(this->m_result);
        this->m_result=nullptr;
    }
    for (auto pair : m_input_value)
    {
        pair.second->clear();
        delete(pair.second);
    }
    this->m_input_value.clear();
    m_to_receive=m_outputs.size();
}
expression::~expression(){
    excute();
}

int expression::add_op_param_double(
    const char *key,
    double value)
{
    printf("[Expression/%d/%s]: Adding param double:%s\n",m_id,m_name.length()==0?m_type.data():(m_type+"/"+m_name).data(),key);
    this->m_params[key]=new tensor(value);
    printf("[Expression/%d/%s]: Added param double:%s\n",m_id,m_name.length()==0?m_type.data():(m_type+"/"+m_name).data(),key);
    return 0;
}

int expression::add_op_param_ndarray(
    const char *key,
    int dim,
    size_t shape[],
    double data[])
{
    printf("[Expression/%d/%s]: Adding param ndarray:%s\n",m_id,m_name.length()==0?m_type.data():(m_type+"/"+m_name).data(),key);
    this->m_params[key]=new tensor(dim, shape, data);
    this->m_params[key]->print();
    printf("[Expression/%d/%s]: Added param ndarray:%s\n",m_id,m_name.length()==0?m_type.data():(m_type+"/"+m_name).data(),key);
    return 0;
}

void expression::calculate(){
    tensor* value=nullptr;
    if (this->m_type.compare("Add")==0){
        value=(*this->m_inputs[0]->getResult()) + (*this->m_inputs[1]->getResult());
    }
    else if (this->m_type.compare("Sub")==0){
        value=(*this->m_inputs[0]->getResult()) - (*this->m_inputs[1]->getResult());
    }
    else if (this->m_type.compare("Mul")==0){
        value=(*this->m_inputs[0]->getResult()) * (*this->m_inputs[1]->getResult());
    }
    else if (this->m_type.compare("Neg")==0){
        
    }
    else if (this->m_type.compare("Const")==0){
        value=this->m_params["value"]->copy();
    }
    else if (this->m_type.compare("Input")==0){
        printf("[Expression/%d/%s]: Unknown error, Input used before value set\n",m_id,m_name.length()==0?m_type.data():(m_type+"/"+m_name).data());
    }
    else if (this->m_type.compare("Input2d")==0){
        printf("[Expression/%d/%s]: Unknown error, Input2D used before value set\n",m_id,m_name.length()==0?m_type.data():(m_type+"/"+m_name).data());
    }
    else if (this->m_type.compare("MaxPool2d")==0){
        /* code */
    }
    else if (this->m_type.compare("ReLu")==0){
        /* code */
    }
    else if (this->m_type.compare("Conv2d")==0){
        /* code */
    }
    else if (this->m_type.compare("Linear")==0){
        /* code */
    }
    set_value(value);
}

void expression::set_value(tensor* value){
    printf("[Expression/%d/%s]: Set value\n",m_id,m_name.length()==0?m_type.data():(m_type+"/"+m_name).data());
    if (this->m_result!=nullptr)
    {
        m_result->clear();
        delete(this->m_result);
        this->m_result=nullptr;
    }
    this->m_result=value;
    this->m_result->print();
}

void expression::post(){
    m_to_receive=m_outputs.size();
    if (this->m_result==nullptr)
    {
        printf("[Expression/%d/%s]: Posting when no result. Will calculate then.\n",m_id,m_name.length()==0?m_type.data():(m_type+"/"+m_name).data());
        calculate();
    }
    for (auto exp: m_outputs)
    {
        exp->receive(m_id, m_result);
    }
}

void expression::receive(int posterID, tensor* value){
    this->m_input_value[posterID]=value;
    //Check if it's time to calculate
    bool cal=true;
    for (auto exp: m_inputs)
    {
        if (m_input_value.count(exp->getID())<=0)
        {
            cal=false;
            break;
        }
    }
    //Calculate and post
    if (cal)
    {
        calculate();
        for (auto exp : m_inputs)
        {
            exp->countdown();
        }
        post();
    }
}

void expression::countdown(){
    --m_to_receive;
    if (this->m_to_receive<=0)
    {
        printf("[Expression/%d/%s]: Countdowned\n",m_id,m_name.length()==0?m_type.data():(m_type+"/"+m_name).data());
        this->clear_runtime();
    }
}

void expression::clear_runtime(){
    if (this->m_result!=nullptr)
    {
        delete(this->m_result);
        this->m_result=nullptr;
    }
    this->m_input_value.clear();
    m_to_receive=m_outputs.size();
}


void expression::clear_all(){
    for (auto exp : m_inputs)
    {
        exp->clear_all();
    }
    clear_runtime();
}

void expression::excute(){
    printf("[Expression/%d/%s]: Excuting\n",m_id,m_name.length()==0?m_type.data():(m_type+"/"+m_name).data());
    for (auto exp : m_inputs)
    {
        exp->excute();
    }
    clear_runtime();
    for (auto pair : m_params)
    {
        pair.second->clear();
        delete(pair.second);
    }
    m_params.clear();
    for (auto exp:m_inputs)
    {
        try
        {
            delete(exp);
        }
        catch(const std::exception& e)
        {
            //do nothing
        }
        
    }
    
}