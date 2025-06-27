


# class QTreeOperator:
#     def __init__(self, qtree: Qtree, kernel):
#         self.domain = qtree
#         self._kernel = kernel

#     @property
#     def eigenvalues(self) -> list(float):
#         cohort = self.domain.Qc
#         c = cohort - np.append([1], cohort[:-1])
#         d = 1 / cohort
    
#         kd = np.array([self._kernel(x) for x in d])
    
#         arr = [
#             d[i] * ( c[:i] @ kd[:i] + kd[i])
#             for i in range(len(self.domain.Q))
#         ]
#         return np.array(arr)

#     def get_time_series(self, delta_t: float, num: int, initial_value: QTreeFunction):
#         offset = 0
#         steps = np.arange(0,num)/num

#         while True:
#             ts = delta_t * (offset + steps)
                
#         pass