import boto3
from datetime import datetime
import pytz

def get_step_function_logs(state_machine_arn, start_date=None, end_date=None):
    """
    Collects the events log for a specific AWS Step Functions state machine.

    :param state_machine_arn: The ARN of the Step Functions state machine
    :param start_date: Optional, filter logs starting from this date (timezone-aware datetime object)
    :param end_date: Optional, filter logs until this date (timezone-aware datetime object)
    """
    # Initialize boto3 client for Step Functions
    stepfunctions_client = boto3.client('stepfunctions')

    try:
        # List executions for the state machine
        executions = stepfunctions_client.list_executions(
            stateMachineArn=state_machine_arn,
            statusFilter='SUCCEEDED'  # You can filter by RUNNING, FAILED, etc.
        )

        print(f"Fetching logs for Step Functions state machine: {state_machine_arn}\n")

        # Iterate through executions
        for execution in executions['executions']:
            execution_arn = execution['executionArn']
            start_time = execution['startDate']
            stop_time = execution['stopDate']
            
            # Ensure start_date and end_date are timezone-aware and in UTC
            if start_date:
                start_date = start_date.astimezone(pytz.utc)
            if end_date:
                end_date = end_date.astimezone(pytz.utc)

            # Filter by start_date and end_date if provided
            if start_date and start_time < start_date:
                continue
            if end_date and stop_time > end_date:
                continue
            
            print(f"Execution ARN: {execution_arn}")
            print(f"Start Time: {start_time}, Stop Time: {stop_time}")

            # Get execution history
            history = stepfunctions_client.get_execution_history(
                executionArn=execution_arn,
                reverseOrder=False  # Set to True if you want events in reverse order
            )

            # Print execution events
            for event in history['events']:
                timestamp = event['timestamp']
                event_type = event['type']
                details = event.get('executionSucceededEventDetails') or \
                          event.get('executionFailedEventDetails') or \
                          event.get('executionStartedEventDetails') or \
                          event.get('stateEnteredEventDetails') or {}
                
                print(f"{timestamp} - {event_type}")
                if details:
                    print(f"  Details: {details}")
            print("-" * 80)

    except stepfunctions_client.exceptions.ResourceNotFoundException:
        print(f"The state machine ARN {state_machine_arn} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    # Replace with your Step Functions state machine ARN
    state_machine_arn = "arn:aws:states:us-east-1:211125778552:stateMachine:DataParallel-CosmicAI"

    # Optional: Define a time range (use timezone-aware UTC datetimes)
    start_date = datetime(2024, 11, 17, 0, 0, 0, tzinfo=pytz.utc)  # Example: Start from this date
    end_date = datetime(2024, 11, 17, 23, 0, 0, tzinfo=pytz.utc)    # Example: Until this date

    get_step_function_logs(state_machine_arn, start_date, end_date)
