import tensorflow as tf

__all__ = ["ModelBase"]

class ModelBase:
    def _ensure_input(self,x):
        """
        Ensure input type and shape

        Parameters
        ----------
        input : Any
            Input values

        Returns
        -------
        x : tf.Tensor
            Input values with shape=(-1,*self.input_shape) and dtype=self.dtype
        """
        x = tf.constant(x,dtype=self.dtype)
        if tf.rank(x) < self.call_rank:
            x = tf.reshape(x,[-1,*self.input_shape.as_list()])
        return x

    def _ensure_output(self,y):
        """
        Ensure output type and shape

        Parameters
        ----------
        y : Any
            Output values

        Returns
        -------
        y : tf.Tensor
           Output values with shape=(-1,self.layers[-1].units) and dtype=self.dtype
        """
        y = tf.constant(y,dtype=self.dtype)
        if tf.rank(y) < self.output_rank:
            y = tf.reshape(y,[-1,self.layers[-1].units])
        return y

