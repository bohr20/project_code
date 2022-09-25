#ifndef EVALUATION_H
#define EVALUATION_H

#include "expression.h"
#include "tensor.h"

class evaluation
{
    private:
        int m_id;
        tensor* m_result;
        expression* m_output;
        std::vector<expression*> m_inputs;
        std::map<std::string, tensor*> m_params;

    public:
        evaluation(expression* output, std::vector<expression*> inputs);
        ~evaluation();
        void clear_runtime();

        int add_kwargs_double(
            const char *key,
            double value);

        int add_kwargs_ndarray(
            const char *key,
            int dim,
            size_t shape[],
            double data[]);

        // return 0 for success
        int execute();

        // return the variable computed by the last expression
        tensor* get_result();


}; // class evaluation

#endif // EVALUATION_H
