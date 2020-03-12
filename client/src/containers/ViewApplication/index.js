import React, { Component } from 'react';
import {connect, Provider} from 'react-redux';
import { createStore, combineReducers } from 'redux';
import { reducer as reduxFormReducer } from 'redux-form';
import LayoutWrapper from '../../components/utility/layoutWrapper';
import PageTitle from '../../components/utility/paperTitle';
import Papersheet, { DemoWrapper } from '../../components/utility/papersheet';
import { FullColumn , HalfColumn, OneThirdColumn, TwoThirdColumn, Row, Column} from '../../components/utility/rowColumn';
import {withStyles} from "@material-ui/core";
import action from "../../redux/ViewApplication/actions";
import {store} from "../../redux/store";
import SingleCard from '../Shuffle/singleCard.js';
import BoxCard from '../../components/boxCard';
import Img7 from '../../images/7.jpg';
import user from '../../images/user.jpg';
import ActionButton from "../../components/mail/singleMail/actionButton";
import {Button} from "../UiElements/Button/button.style";
import Icon from "../../components/uielements/icon";
import moment from 'moment';
import {Chip, Wrapper} from "../UiElements/Chips/chips.style";
import Avatar from "../../components/uielements/avatars";
import {Table} from "../ListApplications/materialUiTables.style";
import {TableBody, TableCell, TableHead, TableRow} from "../../components/uielements/table";

class ViewApplication extends Component {
    state = {
        result: '',
    };

    componentDidMount() {
        store.dispatch(action.requestApplication(this.props.match.params.id));
    }

    launchTestingSequenceButtonClicked() {
        store.dispatch(action.requestNewTestingSequence(this.props.match.params.id));
    }

    render() {
        console.log(this.props.testingSequences);

        const { result } = this.state;
        return (
                this.props.application ?
                    <LayoutWrapper>
                        <FullColumn>
                            <Row>
                                <HalfColumn>
                                            <SingleCard src={Img7} grid/>
                                </HalfColumn>

                                <HalfColumn>
                                    <Papersheet
                                        title={`${this.props.application.name}`}
                                        subtitle={`Last Tested ${moment(this.props.timestamp).format('MMM Do, YYYY')}`}
                                    >
                                        <Row>
                                        <HalfColumn>
                                            <div>
                                                Learning:
                                                <Chip
                                                    avatar={<Icon style={{ fontSize: 22 }}>info-outline</Icon>}
                                                    label="In Progress"
                                                />
                                            </div>
                                        </HalfColumn>
                                        <HalfColumn>
                                            <div>
                                                Testing:
                                                <Chip
                                                    avatar={<Icon style={{ fontSize: 22 }}>info-outline</Icon>}
                                                    label="In Progress"
                                                />
                                            </div>
                                        </HalfColumn>
                                        </Row>
                                        <Row>
                                        <FullColumn>
                                                <DemoWrapper>
                                                    <Button variant="extended" color="primary" onClick={() => this.launchTestingSequenceButtonClicked()}>
                                                        Launch New Testing Sequence
                                                        <Icon className="rightIcon">send</Icon>
                                                    </Button>
                                                </DemoWrapper>
                                        </FullColumn>
                                        </Row>
                                    </Papersheet>
                                </HalfColumn>
                            </Row>


                            <Row>
                                <FullColumn>
                                    <Papersheet title={"Recent Testing Sequences"}>


                                        <Table>
                                            <TableHead>
                                                <TableRow>
                                                    <TableCell>ID</TableCell>
                                                    <TableCell>Status</TableCell>
                                                    <TableCell>Test Start Time</TableCell>
                                                    {/*<TableCell>Test Finish Time</TableCell>*/}
                                                    <TableCell>Bug Found</TableCell>
                                                </TableRow>
                                            </TableHead>
                                            <TableBody>

                                                {(this.props.testingSequences || []).map(testingSequence => {
                                                    return (
                                                        <TableRow key={testingSequence._id.$oid} hover={true} onClick={() => this.props.history.push(`/dashboard/testing_sequences/${testingSequence._id.$oid}`)}>
                                                            <TableCell>{testingSequence._id.$oid}</TableCell>
                                                            <TableCell>{testingSequence.status}</TableCell>
                                                            <TableCell>{testingSequence.startDate}</TableCell>
                                                            {/*<TableCell>{testingSequence.endDate}</TableCell>*/}
                                                            <TableCell>{testingSequence.bugsFound}</TableCell>
                                                        </TableRow>
                                                    );
                                                })}
                                            </TableBody>
                                        </Table>


                                    </Papersheet>
                                </FullColumn>
                            </Row>

                            <Row>
                                <FullColumn>
                                    <Papersheet title={"Recent Training Sequences"}>


                                        <Table>
                                            <TableHead>
                                                <TableRow>
                                                    <TableCell>ID</TableCell>
                                                    <TableCell>Status</TableCell>
                                                    <TableCell>Training Start Time</TableCell>
                                                </TableRow>
                                            </TableHead>
                                            <TableBody>

                                                {(this.props.trainingSequences || []).map(trainingSequence => {
                                                    return (
                                                        <TableRow key={trainingSequence._id.$oid} hover={true} onClick={() => this.props.history.push(`/dashboard/testing_sequences/${trainingSequence._id.$oid}`)}>
                                                            <TableCell>{trainingSequence._id.$oid}</TableCell>
                                                            <TableCell>{trainingSequence.status}</TableCell>
                                                            <TableCell>{trainingSequence.startTime}</TableCell>
                                                        </TableRow>
                                                    );
                                                })}
                                            </TableBody>
                                        </Table>


                                    </Papersheet>
                                </FullColumn>
                            </Row>
                        </FullColumn>
                    </LayoutWrapper>
                    : null

        );
    }
}

const mapStateToProps = (state) => {return { ...state.ViewApplication} };
export default connect(mapStateToProps)(ViewApplication);

