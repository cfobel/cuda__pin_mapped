import os

env = Environment()
env.Tool('cuda')
env.Append(CPPPATH=os.environ['CPATH'].split(':'))
env.Program('pin_mapped.cu', LIBS=['cuda', 'cudart'])
