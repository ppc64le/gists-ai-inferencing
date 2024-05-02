# Deploying the Risk Predictor Model in OpenShift Container Platform

Contributors
----------------------
🎉🌟Thank you to all the contributors who contributed to this project: Revanth Atmakuri, Shadman Kaif, Theresa Xu.🎉🌟

Objective
----------------------
The objective of this project is to deploy the Risk Predictor Model in an OpenShift Container Platform environment, enabling integration and utilization of the model for risk prediction tasks.

Scope
----------------------
This project will involve:

1. Building the image of the Risk Predictor inferencing code using Podman.
2. Pushing the built image into the OpenShift Internal Registry.
3. Retrieving the image stored in the OpenShift Internal Registry.
4. Deploying the model using Kubernetes Deployment on OpenShift.
5. Configuring a service for the deployed model.
6. Setting up an ingress for the model endpoint.
7. Verifying the deployment's functionality and scalability.
   
This project will not cover the development or training of the Risk Predictor Model itself, focusing solely on its deployment within an OpenShift environment.

How-To
----------------------
### Step 1: Build the image of Risk Predictor inferencing code using Podman
* Clone this repository and enter this directory, `cd risk-predictor-workload-in-ocp`
* `podman build -t risk-predictor .`
* `podman images`
* Verify you have an image called localhost/risk-predictor.

### Step 2: Push the build image into the OpenShift Internal Registry
* `podman login -u kubeadmin -p $(oc whoami -t) --tls-verify=false default-route-openshift-image-registry.apps.ai.toropsp.com`
* *podman tag localhost/risk-predictor:latest default-route-openshift-image-registry.apps.ai.toropsp.com/riskpredictor/risk-predictor:latest`
* Make sure you see a new image with this name when run `podman images`
* `podman push default-route-openshift-image-registry.apps.ai.toropsp.com/riskpredictor/risk-predictor --tls-verify=false`

### Step 3: Get the image stored in the OpenShift Internal Registry
* `oc get is`

### Step 4: Deployment 
Make sure you note the full path to the image in step 3, and edit the value called “image” in the deployment file to reflect the image path from step 3.
```
apiVersion: apps/v1
kind: Deployment
metadata:
  name: riskpredictor-endpoint
spec:
  replicas: 64
  selector:
    matchLabels:
      app: riskpredictor-endpoint
  template:
    metadata:
      labels:
        app: riskpredictor-endpoint
    spec:
      containers:
        - name: riskpredictor-endpoint
          image: image-registry.openshift-image-registry.svc:5000/riskpredictor/risk-predictor@sha256:a8fc733ef6948834debaa612fac8ddec309e1ff817db629ddebecdc735a4288f
          imagePullPolicy: IfNotPresent
          ports:
          - containerPort: 5000
          resources:
            limits:
              cpu: 1000m
              memory: 5700Mi #5Gi
            request:
              cpu: 1000m
              memory: 5700Mi #5Gi
```
* Deploy the pods by running `oc apply -f deployment_risk_predictor.yaml`

### Step 5: Deploy the service
```
apiVersion: v1
kind: Service
metadata:
  name: riskpredictor-endpoint
  namespace: riskpredictor
spec:
  ports:
  - port: 5000
    protocol: TCP
    targetPort: 5000
  selector:
    app: riskpredictor-endpoint
  type: NodePort
```
* Apply the changes by running `oc apply -f service_risk_predictor_nodeport.yaml`

### Step 6: Deploy the ingress
```
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: riskpredictor-endpoint-ingress
spec:
  rules:
  - http:
      paths:
      - path: /predict_risk
        pathType: Prefix
        backend:
          service:
            name: riskpredictor-endpoint
            port:
              number: 5000
      - path: /testing
        pathType: Prefix
        backend:
          service:
            name: riskpredictor-endpoint
            port:
              number: 5000
```
* To apply the changes, run `oc apply -f ingress_risk_predictor_nodeport.yaml`

### Step 7: Get the port number
* Run the command `oc get svc`
* An example output is: 
```
#Eg: [root@bastion risk_predictor_workload]# oc get svc
#NAME                     TYPE       CLUSTER-IP      EXTERNAL-IP   PORT(S)          AGE
#riskpredictor-endpoint   NodePort   172.30.247.79   <none>        5000:32479/TCP   45h
```
From this, the port number would be the second port - 32479.

### Step 8: Verify the deployment
* `oc get pod | grep riskpredictor | wc -l`
* Make sure the above output give the number reflecting the number of replicas by running:
  
`curl -X POST -H "Content-Type: application/json" -d '{"example": [37899722504, 81533243, 2, 3, 246.643, 0.05, 15791552, 19366186, "Product #81533243", 10, "medium", 78.2992, 0.0, "10000-50000$", 2500]}' http://ai-w2.ai.toropsp.com:32479/predict_risk`

### Steps to Scale the pods or reallocate the resources
In order to scale the pods, you can open the deployment file and change the values respective to your needs:
```
[root@bastion risk_predictor_workload]# oc get deployment
NAME                     READY   UP-TO-DATE   AVAILABLE   AGE
riskpredictor-endpoint   64/64   64           64          46h
[root@bastion risk_predictor_workload]# oc edit deployment riskpredictor-endpoint
```
Here is the yaml file that's opened which you can change to reflect your resource requirements:
```
# Please edit the object below. Lines beginning with a '#' will be ignored,
# and an empty file will abort the edit. If an error occurs while saving this file will be
# reopened with the relevant failures.
#
apiVersion: apps/v1
kind: Deployment
metadata:
  annotations:
    deployment.kubernetes.io/revision: "27"
    kubectl.kubernetes.io/last-applied-configuration: |
      {"apiVersion":"apps/v1","kind":"Deployment","metadata":{"annotations":{},"name":"riskpredictor-endpoint","namespace":"riskpredictor"},"spec":{"replicas":64,"selector":{"matchLabels":{"app":"riskpredictor-endpoint"}},"template":{"metadata":{"labels":{"app":"riskpredictor-endpoint"}},"spec":{"containers":[{"image":"image-registry.openshift-image-registry.svc:5000/riskpredictor/risk-predictor@sha256:a8fc733ef6948834debaa612fac8ddec309e1ff817db629ddebecdc735a4288f","imagePullPolicy":"IfNotPresent","name":"riskpredictor-endpoint","ports":[{"containerPort":5000}],"resources":{"limits":{"cpu":"1000m","memory":"5700Mi"},"request":{"cpu":"1000m","memory":"5700Mi"}}}]}}}}
  creationTimestamp: "2024-04-30T17:30:39Z"
  generation: 31
  name: riskpredictor-endpoint
  namespace: riskpredictor
  resourceVersion: "3148529"
  uid: 25263d4c-212d-4e98-ab55-0838e17262fe
spec:
  progressDeadlineSeconds: 600
  replicas: 64
  revisionHistoryLimit: 10
  selector:
    matchLabels:
      app: riskpredictor-endpoint
  strategy:
    rollingUpdate:
      maxSurge: 25%
      maxUnavailable: 25%
    type: RollingUpdate
  template:
    metadata:
      creationTimestamp: null
      labels:
        app: riskpredictor-endpoint
    spec:
      containers:
      - image: image-registry.openshift-image-registry.svc:5000/riskpredictor/risk-predictor@sha256:a8fc733ef6948834debaa612fac8ddec309e1ff817db629ddebecdc735a4288f
        imagePullPolicy: IfNotPresent
        name: riskpredictor-endpoint
        ports:
        - containerPort: 5000
          protocol: TCP
        resources:
          limits:
            cpu: "1"
```
