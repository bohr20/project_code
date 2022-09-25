#ifndef EXPRESSION_H
#define EXPRESSION_H

#include <vector>
#include <string.h>
#include <map>
#include "tensor.h"

class evaluation;

class expression
{
    friend class evaluation;

    private:
        //Expression parameters
        int m_id; 
        std::string m_name;
        std::string m_type;
        std::vector<expression*> m_inputs;
        std::vector<expression*> m_outputs;
        std::map<std::string, tensor*>m_params;

        //Runtime staff
        tensor* m_result;
        std::map<int, tensor*>m_input_value;
        size_t m_to_receive; 

    public:
        expression();
        expression(
            int id,
            const char *op_name,
            const char *op_type,
            std::vector<expression*> inputs);
        ~expression();

        int add_op_param_double(
            const char *key,
            double value);

        int add_op_param_ndarray(
            const char *key,
            int dim,
            size_t shape[],
            double data[]);

        //Fetch params
        int getID() const {return m_id;}
        std::string getName() const {return m_name;}
        std::string getType() const {return m_type;}
        std::vector<int> getInputs() const { std::vector<int> ids; for(auto exp:m_inputs)ids.push_back(exp->m_id);return ids;}
        std::vector<int> getOutputs() const { std::vector<int> ids; for(auto exp:m_outputs)ids.push_back(exp->m_id);return ids;}
        std::vector<std::string> getParams() const {std::vector<std::string> paras; for(auto pair:m_params)paras.push_back(pair.first);return paras;}

        tensor *getResult() const {return m_result;}

private:
    void calculate();
    void set_value(tensor* value);
    void post();
    void receive(int posterID, tensor* value);
    void countdown();
    void clear_runtime();
    void clear_all();
    void excute();
}; // class expression

#endif // EXPRESSION_H
