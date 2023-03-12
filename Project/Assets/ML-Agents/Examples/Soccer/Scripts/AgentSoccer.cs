using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Policies;

public enum Team
{
    Blue = 0,
    Purple = 1
}

public class AgentSoccer : Agent
{
    // Note that that the detectable tags are different for the blue and purple teams. The order is
    // * ball
    // * own goal
    // * opposing goal
    // * wall
    // * own teammate
    // * opposing player
    public GameObject ball;
    public GameObject ownGoal;
    public GameObject enemyGoal;
    public GameObject field;

    float w_touch = 0.5f;
    float w_ball_goal = 0.0005f;
    float w_ball_car = 0.0005f;
    //public float w_alignment

    public bool printDebug = false;

    [HideInInspector]
    public Team team;
    float m_KickPower;
    // The coefficient for the reward for colliding with a ball. Set using curriculum.
    float m_BallTouch;

    const float k_Power = 100f;
    float m_Existential;
    float m_ForwardSpeed;
    float forwardVel;


    [HideInInspector]
    public Rigidbody agentRb;
    Rigidbody ballRb;
    SoccerSettings m_SoccerSettings;
    BehaviorParameters m_BehaviorParameters;
    public Vector3 initialPos;
    public float rotSign;

    EnvironmentParameters m_ResetParams;

    public override void Initialize()
    {
        SoccerEnvController envController = GetComponentInParent<SoccerEnvController>();
        if (envController != null)
        {
            m_Existential = 1f / envController.MaxEnvironmentSteps;
        }
        else
        {
            m_Existential = 1f / MaxStep;
        }

        m_BehaviorParameters = gameObject.GetComponent<BehaviorParameters>();
        if (m_BehaviorParameters.TeamId == (int)Team.Blue)
        {
            team = Team.Blue;
            initialPos = new Vector3(transform.position.x - 5f, .5f, transform.position.z);
            rotSign = 1f;
        }
        else
        {
            team = Team.Purple;
            initialPos = new Vector3(transform.position.x + 5f, .5f, transform.position.z);
            rotSign = -1f;
        }
        
        m_ForwardSpeed = 1.0f;
        
        m_SoccerSettings = FindObjectOfType<SoccerSettings>();
        agentRb = GetComponent<Rigidbody>();
        agentRb.maxAngularVelocity = 500;

        ballRb = ball.GetComponent<Rigidbody>();

        m_ResetParams = Academy.Instance.EnvironmentParameters;
    }


    public override void CollectObservations(VectorSensor sensor)
    {   
        // Mirror position observation along x axis for team
        var carPos = gameObject.transform.position - field.transform.position;
        if (m_BehaviorParameters.TeamId != (int)Team.Blue)
        {
            carPos.x = -carPos.x;
        }
        
        sensor.AddObservation(carPos); // Car position on the field
        sensor.AddObservation(agentRb.velocity); // Car velocity
        sensor.AddObservation(gameObject.transform.position - ball.transform.position); // Relative ball position
        sensor.AddObservation(agentRb.velocity - ballRb.velocity); // Relative ball velocity

        var relBall = transform.InverseTransformPoint(ball.transform.position);
        var angle = Mathf.Atan2(relBall.x, relBall.z) / Mathf.PI;
        sensor.AddObservation(angle); // Angle from front of car to ball


        if (printDebug)
        {
            //Debug.Log(gameObject.transform.position - field.transform.position);
        }


        //sensor.AddObservation(gameObject.transform.position); // Car position
        //sensor.AddObservation(agentRb.velocity); // Car velocity
        //sensor.AddObservation(ball.transform.position); // Ball position
        //sensor.AddObservation(ballRb.velocity); // Ball velocity
    }



    public void MoveAgent(ActionBuffers actionBuffers)
    {
        var act = actionBuffers.ContinuousActions;

        var dirToGo = Vector3.zero;
        var rotateDir = Vector3.zero;

        var forwardAxis = Mathf.Clamp(act[0], -1f, 1f);
        var rotateAxis = Mathf.Clamp(act[1], -1f, 1f);

        dirToGo = transform.forward * m_ForwardSpeed * forwardAxis;
        if (forwardAxis < 0f) dirToGo = dirToGo * 0.5f;

        agentRb.AddForce(dirToGo * m_SoccerSettings.agentRunSpeed,
            ForceMode.VelocityChange);
        forwardVel = transform.InverseTransformDirection(agentRb.velocity).z;
        transform.Rotate(transform.up, Time.deltaTime * 10f * forwardVel * rotateAxis);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        var carP = gameObject.transform.position;
        var ballP = ball.transform.position;
        var goalP = enemyGoal.transform.position;

        //Ball distance from enemy goal
        var goal_ball_diff2D = new Vector2(goalP.x, goalP.z) - new Vector2(ballP.x, ballP.z);
        var goal_ball_dist = goal_ball_diff2D.magnitude;
        var gbd = 50.0f;
        AddReward((gbd - Mathf.Min(goal_ball_dist, gbd)) / gbd * w_ball_goal);


        //Car distance from ball
        var car_ball_diff2D = new Vector2(carP.x, carP.z) - new Vector2(ballP.x, ballP.z);
        var car_ball_dist = car_ball_diff2D.magnitude;
        var cbd = 35.0f;
        AddReward(-Mathf.Max(Mathf.Min(car_ball_dist/cbd, 1.0f), 0.2f) * w_ball_car); // Reward negatively the further away the car is from the ball


        if (printDebug)
        {
            Debug.Log(-Mathf.Max(Mathf.Min(car_ball_dist/cbd, 1.0f), 0.2f) * w_ball_car);
        }


        MoveAgent(actionBuffers);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var discreteActionsOut = actionsOut.ContinuousActions;
        //forward
        if (Input.GetKey(KeyCode.W))
        {
            discreteActionsOut[0] = 1;
        }
        if (Input.GetKey(KeyCode.S))
        {
            discreteActionsOut[0] = -1;
        }
        //rotate
        if (Input.GetKey(KeyCode.A))
        {
            discreteActionsOut[1] = -1;
        }
        if (Input.GetKey(KeyCode.D))
        {
            discreteActionsOut[1] = 1;
        }
    }
    /// <summary>
    /// Used to provide a "kick" to the ball.
    /// </summary>
    void OnCollisionEnter(Collision c)
    {
        var force = k_Power * Mathf.Abs(forwardVel);

        if (c.gameObject.CompareTag("ball"))
        {
            AddReward(Mathf.Abs(forwardVel)/25.0f * w_touch);
            var dir = c.contacts[0].point - transform.position;
            dir = dir.normalized;
            c.gameObject.GetComponent<Rigidbody>().AddForce(dir * force);
        }
    }

    public override void OnEpisodeBegin()
    {
        m_BallTouch = m_ResetParams.GetWithDefault("ball_touch", 0);
    }

}
