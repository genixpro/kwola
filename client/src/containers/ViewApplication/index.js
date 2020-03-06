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

    render() {
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
                                                    <Button variant="extended" color="primary">
                                                        Launch New Testing Sequence
                                                        <Icon className="rightIcon">send</Icon>
                                                    </Button>
                                                </DemoWrapper>
                                        </FullColumn>
                                        </Row>
                                    </Papersheet>
                                </HalfColumn>
                            </Row>


                            <Papersheet title={"Recent Testing Sequences"}>


                                <Table>
                                    <TableHead>
                                        <TableRow>
                                            <TableCell>Version</TableCell>
                                            <TableCell>Test Start Time</TableCell>
                                            <TableCell>Test Finish Time</TableCell>
                                            <TableCell>Bug Found</TableCell>
                                        </TableRow>
                                    </TableHead>
                                    <TableBody>

                                        {(this.props.testingSequences || []).map(testingSequence => {
                                            return (
                                                <TableRow key={testingSequence._id.$oid} hover={true} onClick={() => this.props.history.push(`/dashboard/applications/${testingSequence._id.$oid}`)}>
                                                    <TableCell>{testingSequence.name}</TableCell>
                                                    <TableCell>{testingSequence.url}</TableCell>
                                                </TableRow>
                                            );
                                        })}
                                    </TableBody>
                                </Table>





                            </Papersheet>
                        </FullColumn>
                    </LayoutWrapper>
                    : null

        );
    }
}

const mapStateToProps = (state) => {return { ...state.ViewApplication} };
export default connect(mapStateToProps)(ViewApplication);

