using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Policies;

using System;

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

    float w_touch = 5f;
    //float w_ball_goal = 0.0005f;
    float w_ball_car = 0.0005f;
    float w_still = 5f;
    float t_max_still = 4f;
    float timer_still = 0f;
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
    float angle;


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
            initialPos = new Vector3(transform.position.x, transform.position.y, transform.position.z);
            rotSign = 1f;
        }
        else
        {
            team = Team.Purple;
            initialPos = new Vector3(transform.position.x, transform.position.y, transform.position.z);
            rotSign = -1f;
        }
        
        m_ForwardSpeed = 1.0f;
        
        m_SoccerSettings = FindObjectOfType<SoccerSettings>();
        agentRb = GetComponent<Rigidbody>();
        agentRb.maxAngularVelocity = 500;

        ballRb = ball.GetComponent<Rigidbody>();

        timer_still = 0f;

        m_ResetParams = Academy.Instance.EnvironmentParameters;
    }


    public override void CollectObservations(VectorSensor sensor)
    {   
        sensor.Reset();
        /* if (printDebug)
        {
            Debug.Log(sensor);
        } */
        
        // Mirror position observation along x axis for team
        var carPos = gameObject.transform.position - field.transform.position;
        if (m_BehaviorParameters.TeamId != (int)Team.Blue)
        {
            carPos.x = -carPos.x;
        }
        carPos.x = carPos.x / 52.0f;
        carPos.z = carPos.z / 22.0f;
        carPos.y = carPos.y / 10.0f;
        
        sensor.AddObservation(carPos); // Car position on the field
        sensor.AddObservation(transform.InverseTransformDirection(agentRb.velocity) / 35.0f); // Car velocity

        var relPos = gameObject.transform.position - ball.transform.position;
        relPos.x = relPos.x / 104.0f;
        relPos.z = relPos.z / 44.0f;
        relPos.y = relPos.y / 10.0f;
        sensor.AddObservation(relPos); // Relative ball position

        sensor.AddObservation((transform.InverseTransformDirection(agentRb.velocity) - transform.InverseTransformDirection(ballRb.velocity)) / (35.0f*3)); // Relative ball velocity

        var relBall = transform.InverseTransformPoint(ball.transform.position);
        angle = Mathf.Atan2(relBall.x, relBall.z) / Mathf.PI;
        sensor.AddObservation(angle); // Angle from front of car to ball


        if (printDebug)
        {
            //Debug.Log(transform.InverseTransformDirection(agentRb.velocity) - transform.InverseTransformDirection(ballRb.velocity));
        }

        //Rewards
        var carP = gameObject.transform.position;
        var ballP = ball.transform.position;
        var goalP = enemyGoal.transform.position;

        //Ball distance from enemy goal
        /* var goal_ball_diff2D = new Vector2(goalP.x, goalP.z) - new Vector2(ballP.x, ballP.z);
        var goal_ball_dist = goal_ball_diff2D.magnitude;
        var gbd = 50.0f;
        AddReward((gbd - Mathf.Min(goal_ball_dist, gbd)) / gbd * w_ball_goal); */


        //Car distance from ball
        var car_ball_diff2D = new Vector2(carP.x, carP.z) - new Vector2(ballP.x, ballP.z);
        var car_ball_dist = car_ball_diff2D.magnitude;
        var cbd = 35.0f;
        AddReward(-Mathf.Max(Mathf.Min(car_ball_dist/cbd, 1.0f), 0.2f) * w_ball_car); // Reward negatively the further away the car is from the ball


        //Negative reward for staying still too long
        if (Mathf.Abs(forwardVel) < 1)
        {
            timer_still += Time.deltaTime;
            if (timer_still > t_max_still)
            {
                if (printDebug)
                {
                    Debug.Log("Still!");
                }
                AddReward(-w_still);
            }
        } else
        {
            timer_still = 0f;
        }

        if (printDebug)
        {
            //Debug.Log(transform.InverseTransformDirection(agentRb.velocity).z);
            //Debug.Log(-Mathf.Max(Mathf.Min(car_ball_dist/cbd, 1.0f), 0.2f) * w_ball_car);
        }
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

        agentRb.AddForce(dirToGo * m_SoccerSettings.agentRunSpeed, ForceMode.VelocityChange);
        forwardVel = transform.InverseTransformDirection(agentRb.velocity).z;
        transform.Rotate(transform.up, Time.deltaTime * 10f * (forwardVel+1.0f) * rotateAxis);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
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
            AddReward(Mathf.Abs(forwardVel)/35.0f * w_touch);
            var dir = c.contacts[0].point - transform.position;
            dir = dir.normalized;

            var angle_mul = Mathf.Abs((Mathf.Abs(angle) - 0.5f) * 2);
            c.gameObject.GetComponent<Rigidbody>().AddForce(dir * force * angle_mul);
        }
    }

    public override void OnEpisodeBegin()
    {
        m_BallTouch = m_ResetParams.GetWithDefault("ball_touch", 0);
        timer_still = 0f;
    }

}
