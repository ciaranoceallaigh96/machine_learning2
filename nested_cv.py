k_outer = 3
k_inner = 5
X = X
y = y
param_grid = {'speed':[50,100], 'strength':[10,20,30,40]}

model = estimator
outer_scores = []


for i in range(1, k_outer):
  X_outer_train , y_outer_train = cv.split
  X_outer_test, y_outer_test = 
  
  for j in range(1, k_inner):
    X_inner_train, y_inner_train = cv.split
    X_inner_test, y_inner_test = 
    
    for combo in param_grid:
      model.fit(X_train_inner, y_train_inner)
      model.score(X_test_inner, y_test_inner)
      
  best_combo = choose_best_combo 
  model.fit(X_train_outer, y_train_outer)
  result = model.score(X_test_outer, y_test_outer)
  outer_scores.append(result)
    
