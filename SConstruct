import os

env = Environment()
env.Tool('cuda')
env.Append(CPPPATH=os.environ['CPATH'].split(':'))
env.Program('info.cu', LIBS=['cuda', 'cudart'])
