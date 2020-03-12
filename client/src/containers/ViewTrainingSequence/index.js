import React, { Component } from 'react';
import {connect, Provider} from 'react-redux';
import { createStore, combineReducers } from 'redux';
import { reducer as reduxFormReducer } from 'redux-form';
import LayoutWrapper from '../../components/utility/layoutWrapper';
import PageTitle from '../../components/utility/paperTitle';
import Papersheet, { DemoWrapper } from '../../components/utility/papersheet';
import { FullColumn , HalfColumn, OneThirdColumn, TwoThirdColumn, Row, Column} from '../../components/utility/rowColumn';
import {withStyles} from "@material-ui/core";
import action from "../../redux/ViewTrainingSequence/actions";
import {store} from "../../redux/store";
import SingleCard from '../Shuffle/singleCard.js';
import BoxCard from '../../components/boxCard';
import Img7 from '../../images/7.jpg';
import user from '../../images/user.jpg';
import ActionButton from "../../components/mail/singleMail/actionButton";
import {Button} from "../UiElements/Button/button.style";
import Icon from "../../components/uielements/icon";
import moment from 'moment';
import _ from "underscore";
import {Chip, Wrapper} from "../UiElements/Chips/chips.style";
import Avatar from "../../components/uielements/avatars";
import {Table} from "../ListApplications/materialUiTables.style";
import {TableBody, TableCell, TableHead, TableRow} from "../../components/uielements/table";
import { Line } from "react-chartjs-2";

class ViewTrainingSequence extends Component {
    state = {
        result: '',
    };

    componentDidMount() {
        store.dispatch(action.requestTrainingSequence(this.props.match.params.id));
    }


    getConfigurationForHeaderChart() {
        const trainingSequence = this.props.trainingSequence;
        const dataPoints = _.map(trainingSequence.trainingSteps, (step) => step.averageLoss);

        return {
            labels: ["January", "February", "March", "April", "May", "June", "July"],
            datasets: [
                {
                    label: "Average Training Loss",
                    fill: false,
                    lineTension: 0.1,
                    backgroundColor: "rgba(72,166,242,0.6)",
                    borderColor: "rgba(72,166,242,1)",
                    borderCapStyle: "butt",
                    borderDash: [],
                    borderDashOffset: 0.0,
                    borderJoinStyle: "miter",
                    pointBorderColor: "rgba(72,166,242,1)",
                    pointBackgroundColor: "#fff",
                    pointBorderWidth: 1,
                    pointHoverRadius: 5,
                    pointHoverBackgroundColor: "rgba(72,166,242,1)",
                    pointHoverBorderColor: "rgba(72,166,242,1)",
                    pointHoverBorderWidth: 2,
                    pointRadius: 1,
                    pointHitRadius: 10,
                    data: dataPoints
                }
            ]
        };
    }


    render() {
        const { result } = this.state;

        return (
            this.props.trainingSequence ?
                <LayoutWrapper>
                    <FullColumn>
                        <Row>
                            <HalfColumn>
                                <Line data={this.getConfigurationForHeaderChart()} />
                            </HalfColumn>

                            <HalfColumn>
                                <Papersheet
                                    title={`${this.props.trainingSequence._id.$oid}`}
                                    subtitle={}
                                >
                                    <span>Status: {this.props.trainingSequence.status}</span>

                                    <span>Start Time: {moment(this.props.trainingSequence.startTime).format('MMM Do, YYYY')}</span>

                                    {
                                        this.props.trainingSequence.endTime ?
                                            <span>End Time: {moment(this.props.trainingSequence.endTime).format('MMM Do, YYYY')}</span>
                                            : <span>End Time: N/A</span>
                                    }
                                </Papersheet>
                            </HalfColumn>
                        </Row>

                        <Row>
                            <HalfColumn>
                                <Papersheet title={"Testing Sequences"}>
                                    <Table>
                                        <TableHead>
                                            <TableRow>
                                                <TableCell>ID</TableCell>
                                                <TableCell>Status</TableCell>
                                                <TableCell>Test Start Time</TableCell>
                                                <TableCell>Average Reward</TableCell>
                                            </TableRow>
                                        </TableHead>
                                        <TableBody>
                                            {(this.props.testingSequences || []).map(testingSequence => {
                                                return (
                                                    <TableRow key={testingSequence._id.$oid} hover={true}>
                                                        <TableCell>{testingSequence._id.$oid}</TableCell>
                                                        <TableCell>{testingSequence.status}</TableCell>
                                                        <TableCell>{testingSequence.startDate}</TableCell>
                                                        <TableCell>{testingSequence.averageReward}</TableCell>
                                                    </TableRow>
                                                );
                                            })}
                                        </TableBody>
                                    </Table>
                                </Papersheet>
                            </HalfColumn>
                            <HalfColumn>
                                <Papersheet title={"Training Steps"}>
                                    <Table>
                                        <TableHead>
                                            <TableRow>
                                                <TableCell>ID</TableCell>
                                                <TableCell>Status</TableCell>
                                                <TableCell>Step Start Time</TableCell>
                                                <TableCell>Average Loss</TableCell>
                                            </TableRow>
                                        </TableHead>
                                        <TableBody>

                                            {(this.props.trainingSteps || []).map(session => {
                                                return (
                                                    <TableRow key={session._id.$oid} hover={true} onClick={() => this.props.history.push(`/dashboard/execution_sessions/${session._id.$oid}`)}>
                                                        <TableCell>{session._id.$oid}</TableCell>
                                                        <TableCell>{session.status}</TableCell>
                                                        <TableCell>{session.startDate}</TableCell>
                                                        <TableCell>{session.averageLoss}</TableCell>
                                                    </TableRow>
                                                );
                                            })}
                                        </TableBody>
                                    </Table>


                                </Papersheet>
                            </HalfColumn>
                        </Row>

                    </FullColumn>
                </LayoutWrapper>
                : null

        );
    }
}

const mapStateToProps = (state) => {return { ...state.ViewTrainingSequence} };
export default connect(mapStateToProps)(ViewTrainingSequence);

